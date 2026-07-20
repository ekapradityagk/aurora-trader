"""Phase 3 — Merge trades.db + winrate.db into trading.db (single source of truth).

Migrates all data from the old databases into trading.db, then renames
the old files with .bak suffix so nothing breaks mid-flight.
"""
import sqlite3
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "data"
TRADING_DB = BASE / "trading.db"
TRADES_DB = BASE / "trades.db"
WINRATE_DB = BASE / "winrate.db"

def log(msg):
    print(f"  {msg}")

def main():
    print("═══ Phase 3 — Database Migration ═══")

    # ── Connect to master ──
    master = sqlite3.connect(str(TRADING_DB))
    master.row_factory = sqlite3.Row
    master.execute("PRAGMA journal_mode=WAL")

    # ── 1. Migrate trades.db → closed_trades ──
    if TRADES_DB.exists():
        print("\n📦 Migrating trades.db → closed_trades...")
        src = sqlite3.connect(str(TRADES_DB))
        src.row_factory = sqlite3.Row
        try:
            rows = src.execute(
                "SELECT strategy, symbol, side, entry_price, exit_price, "
                "       entry_time, exit_time, pnl, pnl_pct, volume, created_at "
                "FROM trades WHERE pnl IS NOT NULL"
            ).fetchall()
            log(f"Found {len(rows)} trades in trades.db")

            migrated = 0
            for r in rows:
                try:
                    master.execute(
                        """INSERT OR IGNORE INTO closed_trades
                           (symbol, side, entry_price, exit_price, pnl,
                            reason, leverage, closed_at, entry_time, strategy_name)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            r["symbol"] or "—",
                            r["side"] or "LONG",
                            float(r["entry_price"] or 0),
                            float(r["exit_price"] or 0),
                            float(r["pnl"] or 0),
                            "migrated_from_trades_db",
                            1,  # default leverage
                            r["exit_time"] or r["created_at"] or "",
                            r["entry_time"] or "",
                            r["strategy"] or "unknown",
                        ),
                    )
                    migrated += 1
                except Exception as e:
                    log(f"  ⚠️  Skipped row: {e}")
            master.commit()
            log(f"✅ Migrated {migrated} trades from trades.db")
        except Exception as e:
            log(f"  ⚠️  Could not read trades.db: {e}")
        finally:
            src.close()
    else:
        print("\n📦 trades.db not found — skipping")

    # ── 2. Migrate winrate.db trade_results → closed_trades ──
    if WINRATE_DB.exists():
        print("\n📦 Migrating winrate.db trade_results → closed_trades...")
        src = sqlite3.connect(str(WINRATE_DB))
        src.row_factory = sqlite3.Row
        try:
            rows = src.execute(
                "SELECT strategy, symbol, side, entry_price, exit_price, "
                "       pnl, rrr, closed_at "
                "FROM trade_results WHERE pnl IS NOT NULL"
            ).fetchall()
            log(f"Found {len(rows)} trades in winrate.db")

            migrated = 0
            for r in rows:
                try:
                    master.execute(
                        """INSERT OR IGNORE INTO closed_trades
                           (symbol, side, entry_price, exit_price, pnl,
                            reason, leverage, closed_at, entry_time, strategy_name)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            r["symbol"] or "—",
                            r["side"] or "LONG",
                            float(r["entry_price"] or 0),
                            float(r["exit_price"] or 0),
                            float(r["pnl"] or 0),
                            "migrated_from_winrate_db",
                            1,
                            r["closed_at"] or "",
                            "",
                            r["strategy"] or "unknown",
                        ),
                    )
                    migrated += 1
                except Exception as e:
                    log(f"  ⚠️  Skipped row: {e}")
            master.commit()
            log(f"✅ Migrated {migrated} trades from winrate.db")
        except Exception as e:
            log(f"  ⚠️  Could not read winrate.db: {e}")
        finally:
            src.close()
    else:
        print("\n📦 winrate.db not found — skipping")

    # ── 3. Also migrate strategy_selections from trades.db ──
    if TRADES_DB.exists():
        print("\n📦 Migrating strategy_selections → trading.db...")
        src = sqlite3.connect(str(TRADES_DB))
        src.row_factory = sqlite3.Row
        try:
            rows = src.execute(
                "SELECT * FROM strategy_selections"
            ).fetchall()
            if rows:
                # Create table if not exists in master
                master.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_selections (
                        id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        selected_strategy TEXT NOT NULL,
                        version_tag TEXT NOT NULL,
                        market_regime TEXT,
                        regime_confidence REAL,
                        strategy_performance TEXT,
                        reason TEXT,
                        previous_strategy TEXT
                    )
                """)
                migrated = 0
                for r in rows:
                    try:
                        master.execute(
                            """INSERT OR IGNORE INTO strategy_selections
                               (id, timestamp, selected_strategy, version_tag,
                                market_regime, regime_confidence,
                                strategy_performance, reason, previous_strategy)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                r["id"],
                                r["timestamp"],
                                r["selected_strategy"],
                                r["version_tag"],
                                r["market_regime"],
                                r["regime_confidence"],
                                r["strategy_performance"],
                                r["reason"],
                                r["previous_strategy"],
                            ),
                        )
                        migrated += 1
                    except Exception:
                        pass
                master.commit()
                log(f"✅ Migrated {migrated} strategy_selections")
        except Exception:
            log("  No strategy_selections to migrate")
        finally:
            src.close()

    # ── 4. Verify ──
    total = master.execute("SELECT COUNT(*) FROM closed_trades").fetchone()[0]
    log(f"\n📊 Total closed_trades in trading.db: {total}")
    print("\n═══ Migration Complete ═══")
    print("Run 'mv data/trades.db data/trades.db.bak && mv data/winrate.db data/winrate.db.bak' to archive old DBs")

    master.close()

if __name__ == "__main__":
    main()
