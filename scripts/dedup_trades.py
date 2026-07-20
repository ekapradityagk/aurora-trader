"""Deduplicate closed_trades after Phase 3 migration.

Priority: keep rows with real entry_price > 0 first,
then original CB records (not migrated_*), then earliest by id.
"""
import sqlite3
from pathlib import Path

db_path = Path(__file__).resolve().parent.parent / "data" / "trading.db"
conn = sqlite3.connect(str(db_path))
conn.row_factory = sqlite3.Row

# Find duplicates by (symbol, pnl, closed_at)
dup_query = """
SELECT symbol, pnl, closed_at, COUNT(*) as cnt
FROM closed_trades
GROUP BY symbol, ROUND(pnl, 4), closed_at
HAVING cnt > 1
"""
dups = conn.execute(dup_query).fetchall()
print(f"Found {len(dups)} duplicate groups")

removed = 0
for d in dups:
    rows = conn.execute(
        """SELECT id, symbol, entry_price, reason
           FROM closed_trades
           WHERE symbol = ? AND ROUND(pnl, 4) = ? AND closed_at = ?
           ORDER BY
             CASE WHEN entry_price > 0 THEN 0 ELSE 1 END,
             CASE WHEN reason NOT LIKE 'migrated_%' THEN 0 ELSE 1 END,
             id ASC""",
        (d["symbol"], round(d["pnl"], 4), d["closed_at"]),
    ).fetchall()

    # Keep the first (best), delete the rest
    keep_id = rows[0]["id"]
    for r in rows[1:]:
        conn.execute("DELETE FROM closed_trades WHERE id = ?", (r["id"],))
        removed += 1

conn.commit()
print(f"Removed {removed} duplicate rows")

total = conn.execute("SELECT COUNT(*) FROM closed_trades").fetchone()[0]
print(f"Total closed_trades after dedup: {total}")

# Show what we have
print("\n--- Final closed_trades ---")
rows = conn.execute(
    "SELECT strategy_name, symbol, entry_price, exit_price, pnl, reason FROM closed_trades ORDER BY id DESC"
).fetchall()
for r in rows:
    ep = f"{r['entry_price']:.4f}" if r['entry_price'] else "0"
    xp = f"{r['exit_price']:.4f}" if r['exit_price'] else "0"
    pnl = f"{r['pnl']:.4f}" if r['pnl'] else "0"
    print(f"  {r['strategy_name']:20s} {r['symbol']:10s} ep={ep} xp={xp} pnl={pnl}  {r['reason']}")

conn.close()
