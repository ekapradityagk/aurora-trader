# 🤖 Aurora Trader — Architecture & Plan
> New dawn, fresh start. Built for 80%+ win rate.

## Strategy Stack (Blended for 80% WR)

| Strategy | Win Rate | % of Signals | Condition |
|----------|----------|-------------|-----------|
| **Mean Reversion** (BB+RSI) | 71%+ | 60% | ADX < 25 (ranging) |
| **RSI Divergence + SMC** | 75%+ | 25% | At key structure levels |
| **Trend Follow** (Supertrend+EMA) | 50% (3:1 R:R) | 15% | ADX > 25 (trending) |

**Blended target:** 68-75% overall WR. 80% achievable by only taking A+ setups.

## Timeframe Architecture

| Timeframe | Data Source | Frequency | Purpose |
|-----------|-------------|-----------|---------|
| 1m | **WebSocket** | Real-time (<1s) | Price monitoring, stop mgmt |
| 5m | WebSocket | Every 5m close | Fast signal detection |
| 15m | REST poll | Every 15m | Short-term filter |
| **1H** | REST poll | Every 1h | **Primary entry/exit signal** |
| 4H | REST poll | Every 4h | Mid-term structure |
| 1D | REST poll | Every 24h | Trend bias |
| On-chain | API poll | Every 1h | Whale flow overlay |

## Components

### A. Trading Server (24/7)
- Binance WebSocket for real-time klines
- Triple-timeframe confluence check
- ATR-based stop loss → break-even → trailing
- Risk: 1% per trade, 3% daily loss limit
- SQLite trade journal

### B. Learning Server
- Weekly Bayesian hyperopt (Optuna TPE)
- Regime detection (TREND/RANGE/VOLATILE)
- 20-trade rolling WR monitor per strategy
- Auto-switch strategy if WR < 50%

### C. Wallet Scanner
- Exchange flow monitoring (in/out)
- Whale accumulation tracking (1000+ BTC wallets)
- Funding rate analysis
- Stablecoin reserve ratio
- 2+ signals = bias override

### D. Integration Layer
- Git versioning per strategy change
- SQLite winrate log per version
- Rollback to last working version
- GitHub push on every strategy update

## Project Structure

```
~/Developer/aurora-trader/
├── trading_server/
│   ├── server.py              # Main event loop (ASGI)
│   ├── strategies/
│   │   ├── base.py
│   │   ├── mean_reversion.py
│   │   ├── rsi_divergence.py
│   │   └── trend_follow.py
│   ├── exchange/
│   │   ├── __init__.py
│   │   ├── binance_ws.py      # WebSocket streams
│   │   └── binance_rest.py    # REST API calls
│   └── risk/
│       ├── manager.py         # Position sizing + ATR stops/break-even/trailing
│       └── circuit_breaker.py # Daily loss limit
├── learning_server/
│   ├── hyperopt.py            # Optuna optimization
│   ├── regime.py              # Market regime detector
│   ├── analyzer.py            # Trade history analysis
│   └── strategy_selector.py   # Auto-switch logic
├── wallet_scanner/
│   ├── scanner.py             # Main scanner loop
│   ├── exchange_flow.py       # CEX inflow/outflow
│   ├── whale_tracker.py       # Whale wallet monitoring
│   ├── funding_rate.py        # Perpetual funding analysis
│   └── signal_aggregator.py   # Combine signals
├── integration/
│   ├── version_control.py     # Git auto-commit + tags
│   ├── winrate_db.py          # SQLite per-version logging
│   └── rollback.py            # Safe fallback
├── shared/
│   ├── config.py              # Central config
│   ├── logger.py              # Structured logging
│   └── models.py              # Data models
├── data/
│   ├── trades.db              # SQLite trade journal
│   └── winrate.db             # SQLite winrate history
├── logs/
├── config.yaml                # User-editable config
├── requirements.txt
└── README.md
```

## Infrastructure

| Service | Port | Watchdog | Strategy |
|---------|------|----------|----------|
| Aurora Trading Server | 8900 | Hermes cron | Server auto-restart |
| Learning Server | 8901 | Hermes cron | Weekly run |
| Wallet Scanner | 8902 | Hermes cron | Every 1h |

## Versioning Flow

```
Strategy change detected
    ↓
Git commit + tag (v1.0, v1.1, ...)
    ↓
Push to GitHub
    ↓
Deploy new strategy
    ↓
SQLite logs WR per version
    ↓
If WR < 50% over 20 trades → auto-rollback
