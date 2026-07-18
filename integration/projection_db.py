"""Aurora Trader — Projection Profile Database.

Stores saved projection calculator profiles (name, description, config)
in a SQLite table within the integration database.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiosqlite

from shared.logger import get_logger

logger = get_logger("integration.projection_db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS projection_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT DEFAULT '',
    config TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS projection_inputs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL UNIQUE,
    actual_pnl REAL NOT NULL,
    notes TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class ProjectionDB:
    """Persistent storage for projection calculator profiles."""

    def __init__(self, db_path: str = "data/integration.db"):
        self._db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None
        self._log = logger

    async def initialize(self) -> None:
        """Open DB connection and ensure schema exists."""
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(SCHEMA)
        await self._conn.commit()
        self._log.info(f"ProjectionDB initialised: {self._db_path}")

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def save_profile(
        self, name: str, config: Dict[str, Any], description: str = ""
    ) -> int:
        """Save a new profile. Returns the profile ID.

        Raises ValueError if name already exists.
        """
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        config_json = json.dumps(config)
        try:
            cursor = await self._conn.execute(
                """INSERT INTO projection_profiles (name, description, config, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (name, description, config_json, now, now),
            )
            await self._conn.commit()
            return cursor.lastrowid
        except aiosqlite.IntegrityError:
            raise ValueError(f"Profile '{name}' already exists")

    async def update_profile(
        self, profile_id: int, name: str, config: Dict[str, Any], description: str = ""
    ) -> bool:
        """Update an existing profile."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        config_json = json.dumps(config)
        cursor = await self._conn.execute(
            """UPDATE projection_profiles
               SET name = ?, description = ?, config = ?, updated_at = ?
               WHERE id = ?""",
            (name, description, config_json, now, profile_id),
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    async def list_profiles(self) -> List[Dict[str, Any]]:
        """List all saved profiles with their metadata."""
        cursor = await self._conn.execute(
            "SELECT id, name, description, config, created_at, updated_at "
            "FROM projection_profiles ORDER BY updated_at DESC"
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "description": row["description"],
                "config": json.loads(row["config"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    async def get_profile(self, profile_id: int) -> Optional[Dict[str, Any]]:
        """Get a single profile by ID."""
        cursor = await self._conn.execute(
            "SELECT id, name, description, config, created_at, updated_at "
            "FROM projection_profiles WHERE id = ?",
            (profile_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "name": row["name"],
            "description": row["description"],
            "config": json.loads(row["config"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    async def delete_profile(self, profile_id: int) -> bool:
        """Delete a profile by ID."""
        cursor = await self._conn.execute(
            "DELETE FROM projection_profiles WHERE id = ?",
            (profile_id,),
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Real PnL Inputs
    # ------------------------------------------------------------------

    async def save_input(self, date: str, actual_pnl: float, notes: str = "") -> dict:
        """Save or update a real PnL input for a specific date.

        Returns the saved record.
        """
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        await self._conn.execute(
            """INSERT INTO projection_inputs (date, actual_pnl, notes, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(date) DO UPDATE SET
                   actual_pnl = excluded.actual_pnl,
                   notes = excluded.notes,
                   updated_at = excluded.updated_at""",
            (date, actual_pnl, notes, now, now),
        )
        await self._conn.commit()

        cursor = await self._conn.execute(
            "SELECT * FROM projection_inputs WHERE date = ?", (date,)
        )
        row = await cursor.fetchone()
        return {
            "id": row["id"],
            "date": row["date"],
            "actual_pnl": row["actual_pnl"],
            "notes": row["notes"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    async def get_inputs_in_range(self, start_date: str, end_date: str) -> list[dict]:
        """Get all real PnL inputs within a date range."""
        cursor = await self._conn.execute(
            "SELECT * FROM projection_inputs WHERE date >= ? AND date <= ? ORDER BY date ASC",
            (start_date, end_date),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": row["id"],
                "date": row["date"],
                "actual_pnl": row["actual_pnl"],
                "notes": row["notes"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    async def delete_input(self, date: str) -> bool:
        """Delete a real PnL input for a specific date."""
        cursor = await self._conn.execute(
            "DELETE FROM projection_inputs WHERE date = ?", (date,)
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    async def get_all_inputs(self) -> list[dict]:
        """Get all real PnL inputs across all dates."""
        cursor = await self._conn.execute(
            "SELECT * FROM projection_inputs ORDER BY date ASC"
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": row["id"],
                "date": row["date"],
                "actual_pnl": row["actual_pnl"],
                "notes": row["notes"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]
