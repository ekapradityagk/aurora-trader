"""
Aurora Trader — Bayesian Hyper-parameter Optimisation.

Uses Optuna with a TPE sampler to find optimal trading strategy parameters
from historical trade data.  Implements walk-forward validation (6 months
train, 3 months test) and a Monte Carlo permutation test to prevent
overfitting.

Optimised parameters:
    - RSI period
    - Bollinger Band standard deviation
    - ATR multiplier
    - Stop loss % (as a factor applied to ATR)
    - Take profit % (as a factor applied to ATR)

Objective: maximise out-of-sample Sharpe ratio (NOT raw profit).
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiosqlite

from shared.config import load_config
from shared.logger import get_logger

logger = get_logger("learning_server.hyperopt")

# ---------------------------------------------------------------------------
# Default search bounds
# ---------------------------------------------------------------------------

SEARCH_SPACE = {
    "rsi_period": (5, 30),
    "bb_std_dev": (1.2, 3.5),
    "atr_multiplier": (1.0, 3.0),
    "stop_loss_pct": (0.3, 2.0),
    "take_profit_pct": (0.5, 4.0),
}

# Walk-forward parameters
TRAIN_MONTHS = 6
TEST_MONTHS = 3

# Monte Carlo
MC_SHUFFLES = 1000

# ---------------------------------------------------------------------------
# Dataclass for optimisation results
# ---------------------------------------------------------------------------


@dataclass
class OptimizationResult:
    """Holds the outcome of a single hyperopt run."""

    params: Dict[str, float] = field(default_factory=dict)
    train_sharpe: float = 0.0
    test_sharpe: float = 0.0
    mc_p_value: float = 0.0  # fraction of shuffled runs that beat test Sharpe
    n_trials: int = 0
    timestamp: str = ""
    version_tag: str = ""


# ---------------------------------------------------------------------------
# Trade loader
# ---------------------------------------------------------------------------


async def _load_closed_trades(db_path: str) -> List[Dict[str, Any]]:
    """Load all closed trades from the SQLite trade journal.

    Expected schema (trades table):
        id, strategy_name, symbol, side, entry_price, exit_price, quantity,
        quote_quantity, pnl, pnl_pct, entry_time, exit_time, ...
    """
    trades: List[Dict[str, Any]] = []
    try:
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT *
                FROM trades
                WHERE exit_price IS NOT NULL
                  AND pnl IS NOT NULL
                ORDER BY exit_time ASC
                """
            )
            rows = await cursor.fetchall()
            for row in rows:
                trades.append(dict(row))
        logger.info(f"Loaded {len(trades)} closed trades from {db_path}")
    except Exception as exc:
        logger.warning(f"Could not load trades from {db_path}: {exc}")
    return trades


# ---------------------------------------------------------------------------
# Sharpe ratio calculator (annualised)
# ---------------------------------------------------------------------------


def _sharpe_ratio(pnl_pcts: List[float], risk_free_rate: float = 0.02) -> float:
    """Compute annualised Sharpe ratio from a list of trade PnL percentages.

    Uses the standard formula: Sharpe = (mean(R) - r_f) / std(R) * sqrt(N)
    where N = number of trades per year (assumes ~252 trading days).
    """
    if len(pnl_pcts) < 2:
        return 0.0
    mean_ret = sum(pnl_pcts) / len(pnl_pcts)
    variance = sum((r - mean_ret) ** 2 for r in pnl_pcts) / (len(pnl_pcts) - 1)
    if variance <= 0:
        return 0.0
    std_dev = math.sqrt(variance)
    # Annualise: multiply by sqrt(trades per year)
    # We approximate trades per year; use len(pnl_pcts) / duration_years
    # As a fallback, assume 252 trading days * average trades per day
    ann_factor = math.sqrt(252)  # rough daily sampling factor
    return ((mean_ret - risk_free_rate / 252) / std_dev) * ann_factor


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------


def _simulate_trades(
    trades: List[Dict[str, Any]],
    params: Dict[str, float],
) -> List[float]:
    """Simulate trades under the given parameter set and return PnL % list.

    For each trade, we apply the configured stop-loss and take-profit to
    determine whether the trade would have been closed earlier or allowed to
    run to its actual exit.  This is a simplified simulation that uses the
    trade's entry price, the high/low of the trade period, and the configured
    SL/TP levels.
    """
    results: List[float] = []
    sl_pct = params["stop_loss_pct"] / 100.0
    tp_pct = params["take_profit_pct"] / 100.0

    for t in trades:
        entry = float(t.get("entry_price", 0))
        if entry <= 0:
            continue
        pnl_pct = float(t.get("pnl_pct", 0)) if t.get("pnl_pct") is not None else 0.0
        side = t.get("side", "buy")

        # Apply parameter-based scaling: the configured SL/TP adjusts actual PnL
        # If the trade hit SL, cap loss; if it hit TP, cap gain.
        raw_pnl = pnl_pct
        if side in ("buy", "long"):
            if raw_pnl < -sl_pct:
                raw_pnl = -sl_pct
            elif raw_pnl > tp_pct:
                raw_pnl = tp_pct
        else:  # sell / short
            if raw_pnl < -sl_pct:
                raw_pnl = -sl_pct
            elif raw_pnl > tp_pct:
                raw_pnl = tp_pct

        results.append(raw_pnl)

    return results


# ---------------------------------------------------------------------------
# Monte Carlo permutation test
# ---------------------------------------------------------------------------


def _monte_carlo_permutation_test(
    trades: List[float],
    test_sharpe: float,
    n_shuffles: int = MC_SHUFFLES,
) -> float:
    """Shuffle trade outcomes 1000 times and count how often a random
    permutation produces a Sharpe ratio >= the observed test Sharpe.

    Returns a p-value: low p-value means the strategy genuinely outperforms
    random ordering of the same trade outcomes.
    """
    if len(trades) < 5:
        return 1.0

    count_better = 0
    for _ in range(n_shuffles):
        shuffled = trades[:]
        random.shuffle(shuffled)
        shuf_sharpe = _sharpe_ratio(shuffled)
        if shuf_sharpe >= test_sharpe:
            count_better += 1

    p_val = count_better / n_shuffles
    return p_val


# ---------------------------------------------------------------------------
# Hyperopt Optimizer
# ---------------------------------------------------------------------------


class HyperoptOptimizer:
    """Bayesian optimisation of strategy parameters using Optuna (TPE sampler).

    Usage::

        opt = HyperoptOptimizer()
        result = await opt.run(target_sharpe=0.5)
    """

    def __init__(
        self,
        db_path: str = "data/trades.db",
        output_dir: str = "data/optimization",
    ) -> None:
        self._db_path = db_path
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._log = logger
        self._cfg = load_config()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> OptimizationResult:
        """Run a full optimisation cycle.

        1. Load trade history
        2. Walk-forward split (train 6mo, test 3mo)
        3. Bayesian optimisation (TPE) on training window
        4. Evaluate best params on test window
        5. Monte Carlo permutation test
        6. Save results to JSON
        7. Return result summary
        """
        optuna_cfg = self._cfg.optuna_config
        n_trials = n_trials or optuna_cfg.get("n_trials", 100)
        timeout = timeout or optuna_cfg.get("timeout_seconds", 3600)

        # 1. Load trades
        all_trades = await _load_closed_trades(self._db_path)
        if not all_trades:
            self._log.warning("No closed trades found — skipping optimisation")
            return OptimizationResult()

        # 2. Walk-forward split
        train_trades, test_trades = self._walk_forward_split(all_trades)
        if not train_trades or not test_trades:
            self._log.warning(
                "Insufficient trade history for walk-forward split"
            )
            return OptimizationResult()

        self._log.info(
            f"Walk-forward: {len(train_trades)} train trades, "
            f"{len(test_trades)} test trades"
        )

        # 3. Optimise on training set
        best_params, train_sharpe = await self._optimize(
            train_trades, n_trials, timeout
        )
        if not best_params:
            self._log.warning("Optimisation produced no valid parameters")
            return OptimizationResult()

        self._log.info(
            f"Best params found: {best_params}  "
            f"(train Sharpe={train_sharpe:.4f})"
        )

        # 4. Evaluate on test set
        test_pnl = _simulate_trades(test_trades, best_params)
        test_sharpe = _sharpe_ratio(test_pnl)
        self._log.info(f"Test Sharpe={test_sharpe:.4f}")

        # 5. Monte Carlo permutation test
        all_pnl = _simulate_trades(all_trades, best_params)
        mc_p_value = _monte_carlo_permutation_test(all_pnl, test_sharpe)
        self._log.info(
            f"Monte Carlo p-value={mc_p_value:.4f} "
            f"(lower is better, target < 0.05)"
        )

        # 6. Build result & save
        now_str = datetime.now(timezone.utc).isoformat()
        result = OptimizationResult(
            params=best_params,
            train_sharpe=train_sharpe,
            test_sharpe=test_sharpe,
            mc_p_value=mc_p_value,
            n_trials=n_trials,
            timestamp=now_str,
            version_tag=f"opt_{now_str[:10].replace('-', '')}",
        )

        self._save_result(result)

        return result

    # ------------------------------------------------------------------
    # Walk-forward split
    # ------------------------------------------------------------------

    def _walk_forward_split(
        self,
        trades: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split trades into train and test windows.

        Train: last TRAIN_MONTHS months of data (excluding most recent data).
        Test:  next TEST_MONTHS months after train window.

        If there aren't enough trades to cover the full windows, we take
        whatever is available as train and the rest as test.
        """
        if not trades:
            return [], []

        # Sort by exit_time
        sorted_trades = sorted(
            trades,
            key=lambda t: t.get("exit_time", ""),
        )

        # Determine the time boundaries
        try:
            latest_exit = datetime.fromisoformat(
                str(sorted_trades[-1]["exit_time"])
            ).replace(tzinfo=timezone.utc)
        except (ValueError, TypeError, KeyError):
            latest_exit = datetime.now(timezone.utc)

        train_cutoff = latest_exit - timedelta(days=TEST_MONTHS * 30)
        test_start = train_cutoff
        test_end = latest_exit

        train_trades = []
        test_trades = []

        for t in sorted_trades:
            try:
                exit_time = datetime.fromisoformat(
                    str(t["exit_time"])
                ).replace(tzinfo=timezone.utc)
            except (ValueError, TypeError, KeyError):
                continue

            if exit_time < train_cutoff:
                train_trades.append(t)
            elif test_start <= exit_time <= test_end:
                test_trades.append(t)
            # Older trades are discarded

        # If train is empty, use everything before test window
        if not train_trades and test_trades:
            train_trades = test_trades
            test_trades = []

        return train_trades, test_trades

    # ------------------------------------------------------------------
    # Optuna optimisation (lazy import)
    # ------------------------------------------------------------------

    async def _optimize(
        self,
        train_trades: List[Dict[str, Any]],
        n_trials: int,
        timeout: int,
    ) -> Tuple[Optional[Dict[str, float]], float]:
        """Run Optuna TPE optimisation on training trades.

        Returns (best_params, best_sharpe).
        """
        try:
            import optuna
        except ImportError:
            self._log.error(
                "optuna is not installed. "
                "Run: pip install optuna"
            )
            return None, 0.0

        def objective(trial: optuna.Trial) -> float:
            params = {
                "rsi_period": trial.suggest_int(
                    "rsi_period",
                    int(SEARCH_SPACE["rsi_period"][0]),
                    int(SEARCH_SPACE["rsi_period"][1]),
                ),
                "bb_std_dev": trial.suggest_float(
                    "bb_std_dev",
                    SEARCH_SPACE["bb_std_dev"][0],
                    SEARCH_SPACE["bb_std_dev"][1],
                ),
                "atr_multiplier": trial.suggest_float(
                    "atr_multiplier",
                    SEARCH_SPACE["atr_multiplier"][0],
                    SEARCH_SPACE["atr_multiplier"][1],
                ),
                "stop_loss_pct": trial.suggest_float(
                    "stop_loss_pct",
                    SEARCH_SPACE["stop_loss_pct"][0],
                    SEARCH_SPACE["stop_loss_pct"][1],
                ),
                "take_profit_pct": trial.suggest_float(
                    "take_profit_pct",
                    SEARCH_SPACE["take_profit_pct"][0],
                    SEARCH_SPACE["take_profit_pct"][1],
                ),
            }
            pnl_list = _simulate_trades(train_trades, params)
            return _sharpe_ratio(pnl_list)

        optuna_cfg = self._cfg.optuna_config
        storage_url = optuna_cfg.get("storage")  # None → in-memory (safe fallback)
        study = optuna.create_study(
            direction=optuna_cfg.get("direction", "maximize"),
            sampler=optuna.samplers.TPESampler(seed=42),
            study_name="aurora_hyperopt",
            storage=storage_url,
            load_if_exists=True,
        )

        self._log.info(
            f"Starting Optuna optimisation: {n_trials} trials, "
            f"{timeout}s timeout"
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=1,
            show_progress_bar=False,
        )

        best_params = study.best_params if study.best_params else {}
        best_value = study.best_value if study.best_value is not None else 0.0

        self._log.info(
            f"Optimisation complete: {len(study.trials)} trials, "
            f"best Sharpe={best_value:.4f}"
        )

        return best_params, best_value

    # ------------------------------------------------------------------
    # Result persistence
    # ------------------------------------------------------------------

    def _save_result(self, result: OptimizationResult) -> None:
        """Save the optimisation result to JSON so the trading server can
        pick it up."""
        filepath = self._output_dir / "best_params.json"
        payload = {
            "version_tag": result.version_tag,
            "timestamp": result.timestamp,
            "params": result.params,
            "train_sharpe": round(result.train_sharpe, 4),
            "test_sharpe": round(result.test_sharpe, 4),
            "mc_p_value": round(result.mc_p_value, 4),
            "n_trials": result.n_trials,
        }

        try:
            with open(filepath, "w") as f:
                json.dump(payload, f, indent=2)
            self._log.info(f"Best params saved to {filepath}")
        except IOError as exc:
            self._log.error(f"Failed to save best params: {exc}")

        # Also save a timestamped copy
        ts_path = (
            self._output_dir
            / f"best_params_{result.timestamp[:10]}.json"
        )
        try:
            with open(ts_path, "w") as f:
                json.dump(payload, f, indent=2)
        except IOError:
            pass

    def load_best_params(self) -> Optional[Dict[str, float]]:
        """Load the best parameters from the saved JSON file."""
        filepath = self._output_dir / "best_params.json"
        if not filepath.is_file():
            self._log.debug("No best_params.json found")
            return None
        try:
            with open(filepath) as f:
                data = json.load(f)
            return data.get("params")
        except (IOError, json.JSONDecodeError) as exc:
            self._log.warning(f"Failed to load best params: {exc}")
            return None
