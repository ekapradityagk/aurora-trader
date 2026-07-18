#!/usr/bin/env python3
"""
Trailing Stop Watchdog — polls the trading server for new trailing events
and outputs Discord-formatted messages for Hermes cron delivery.

Usage (no_agent=True):
    python3 trailing_watchdog.py

Outputs nothing if no new events (cron stays silent).
Outputs one or more messages separated by --- if new events found.
"""

import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone

# --- Config ---
TRADING_API = "http://127.0.0.1:8900/api/events/trailing"
SEEN_FILE = os.path.expanduser("~/.hermes/scripts/trailing_seen.json")
os.makedirs(os.path.dirname(SEEN_FILE), exist_ok=True)


def fetch_events(since: str = "") -> list[dict] | None:
    """Fetch trailing events from the trading server."""
    url = TRADING_API
    if since:
        url += f"?since={since}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            return data.get("events", [])
    except (urllib.error.URLError, ConnectionError, OSError, json.JSONDecodeError) as e:
        print(f"⚠️ Watchdog: Can't reach trading server — {e}", file=sys.stderr)
        return None  # Signal "try again later" vs "no events"


def load_seen() -> set:
    """Load set of already-reported event keys."""
    try:
        with open(SEEN_FILE) as f:
            return set(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def save_seen(seen: set):
    """Persist reported event keys."""
    with open(SEEN_FILE, "w") as f:
        json.dump(sorted(seen)[-200:], f)  # keep last 200


def event_key(event: dict) -> str:
    """Unique key for dedup — symbol + type + timestamp."""
    return f"{event.get('symbol', '?')}|{event.get('type', '?')}|{event.get('timestamp', '?')}"


def format_message(event: dict) -> str:
    """Format a trailing event as a Discord-friendly message."""
    symbol = event.get("symbol", "???")
    ev_type = event.get("type", "unknown")
    entry = event.get("entry_price", 0)
    curr = event.get("current_price", 0)
    sl = event.get("stop_loss", 0)
    lev = event.get("leverage", 20)

    pnl_pct = ((curr - entry) / entry * lev * 100) if entry else 0
    sl_pct = ((sl - entry) / entry * lev * 100) if entry else 0

    icon = "🔴" if ev_type == "activated" else "🔶"
    title = "Trailing Activated" if ev_type == "activated" else "Trailing Updated"

    msg = (
        f"{icon} **{title} — {symbol}**\n"
        f"Entry: `${entry:.4f}` | Current: `${curr:.4f}`\n"
        f"Trail Stop: `${sl:.4f}` | ROI: `{pnl_pct:+.2f}%`\n"
        f"Locked Profit: `{sl_pct:+.2f}%` | Leverage: `{lev}x`\n"
    )
    return msg


def main():
    seen = load_seen()
    events = fetch_events()

    if events is None:
        # Server unreachable — skip this tick silently
        sys.exit(0)

    new_messages = []
    for ev in events:
        key = event_key(ev)
        if key not in seen:
            seen.add(key)
            new_messages.append(format_message(ev))

    if new_messages:
        save_seen(seen)
        # Print all messages separated by a divider
        print("\n---\n".join(new_messages))
    # Else: silent — nothing to report


if __name__ == "__main__":
    main()
