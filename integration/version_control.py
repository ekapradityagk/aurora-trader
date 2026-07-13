"""
Aurora Trader — Git-Based Strategy Versioning.

Provides automatic versioning of strategy parameters using git:
  - Auto-commits when strategy parameters change
  - Creates semantic version tags (v1.0.0, v1.1.0, etc.)
  - Stores version metadata in SQLite
  - Initialises a git repo if none exists
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from shared.config import load_config
from shared.logger import get_logger

logger = get_logger("integration.version_control")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_DB_PATH = "data/integration.db"
_TAG_PREFIX = "v"


# ---------------------------------------------------------------------------
# VersionController
# ---------------------------------------------------------------------------


class VersionController:
    """Manages semantic versioning of strategy parameters with git integration.

    Usage::

        vc = VersionController()
        tag = vc.save_version("ema_crossover", {"ema_fast": 9, "ema_slow": 50})
        versions = vc.list_versions()
        params = vc.get_version("v1.2.3")
    """

    def __init__(self, project_root: Optional[str] = None, db_path: Optional[str] = None) -> None:
        self._cfg = load_config()
        self._log = logger

        # Resolve project root
        if project_root:
            self._project_root = Path(project_root).resolve()
        else:
            # Walk up from integration/ to find the project root
            self._project_root = Path(__file__).resolve().parent.parent

        # Resolve DB path
        if db_path:
            self._db_path = Path(db_path)
        else:
            db_rel = self._cfg.data.get("integration", {}).get("database", {}).get("path", _DEFAULT_DB_PATH)
            self._db_path = self._project_root / db_rel

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._ensure_git_repo()

        # Config for git behaviour
        git_cfg = self._cfg.data.get("integration", {}).get("git", {})
        self._auto_commit = git_cfg.get("auto_commit", True)
        self._auto_tag = git_cfg.get("auto_tag", True)
        self._tag_prefix = git_cfg.get("tag_prefix", _TAG_PREFIX)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_version(self, strategy_name: str, params_dict: Dict[str, Any]) -> str:
        """Commit current parameters to git and create a semantic version tag.

        Args:
            strategy_name: Name of the strategy (e.g. 'ema_crossover').
            params_dict: The parameter dict to version.

        Returns:
            The semantic version tag string (e.g. 'v1.0.0').
        """
        # 1. Compute a hash of the parameters for change detection
        params_hash = self._hash_params(params_dict)

        # 2. Check if we already have this exact params hash for this strategy
        existing = self._get_version_by_hash(strategy_name, params_hash)
        if existing:
            self._log.info(
                f"Parameters for '{strategy_name}' unchanged (hash={params_hash[:12]}...) "
                f"— returning existing tag {existing}"
            )
            return existing

        # 3. Bump version
        new_tag = self._bump_version(strategy_name)

        # 4. Git operations
        git_commit_hash = ""
        try:
            git_commit_hash = self._git_commit_and_tag(strategy_name, params_dict, new_tag)
        except Exception as exc:
            self._log.warning(f"Git operation failed (non-fatal): {exc}")

        # 5. Persist metadata
        self._insert_version(strategy_name, new_tag, params_hash, params_dict, git_commit_hash)

        self._log.info(
            f"Saved version '{new_tag}' for strategy '{strategy_name}' "
            f"(hash={params_hash[:12]}..., commit={git_commit_hash[:12] if git_commit_hash else 'none'})"
        )
        return new_tag

    def list_versions(self) -> List[Dict[str, Any]]:
        """Return all saved versions with metadata, newest first."""
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "SELECT version_tag, strategy_name, params_hash, deployed_at, "
                "git_commit_hash, description "
                "FROM strategy_versions ORDER BY rowid DESC"
            )
            rows = cursor.fetchall()
            return [
                {
                    "version_tag": row[0],
                    "strategy_name": row[1],
                    "params_hash": row[2],
                    "deployed_at": row[3],
                    "git_commit_hash": row[4],
                    "description": row[5],
                }
                for row in rows
            ]
        finally:
            conn.close()

    def get_version(self, tag: str) -> Optional[Dict[str, Any]]:
        """Return the full version record (including params) for a given tag.

        Args:
            tag: Version tag string (e.g. 'v1.0.0').

        Returns:
            Dict with version metadata and the saved ``parameters``, or None.
        """
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "SELECT version_tag, strategy_name, params_hash, deployed_at, "
                "git_commit_hash, description, parameters "
                "FROM strategy_versions WHERE version_tag = ?",
                (tag,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return {
                "version_tag": row[0],
                "strategy_name": row[1],
                "params_hash": row[2],
                "deployed_at": row[3],
                "git_commit_hash": row[4],
                "description": row[5],
                "parameters": json.loads(row[6]) if row[6] else {},
            }
        finally:
            conn.close()

    def get_latest_version(self, strategy_name: str) -> Optional[str]:
        """Return the latest tag for a given strategy, or None."""
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "SELECT version_tag FROM strategy_versions "
                "WHERE strategy_name = ? ORDER BY rowid DESC LIMIT 1",
                (strategy_name,),
            )
            row = cursor.fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Git helpers
    # ------------------------------------------------------------------

    def _ensure_git_repo(self) -> None:
        """Initialise a git repository at the project root if one doesn't exist."""
        git_dir = self._project_root / ".git"
        if not git_dir.is_dir():
            self._log.info(f"No git repo found — initialising at {self._project_root}")
            try:
                subprocess.run(
                    ["git", "init"],
                    cwd=str(self._project_root),
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30,
                )
                # Create a .gitignore if not present
                gitignore = self._project_root / ".gitignore"
                if not gitignore.exists():
                    gitignore.write_text(
                        "__pycache__/\n*.pyc\n*.pyo\n.env\nvenv/\n.venv/\n"
                        "data/*.db\nlogs/\n*.log\n"
                    )
                # Initial commit if the repo is empty
                status = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=str(self._project_root),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if status.stdout.strip():
                    subprocess.run(
                        ["git", "add", "-A"],
                        cwd=str(self._project_root),
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    subprocess.run(
                        ["git", "commit", "-m", "Initial commit — Aurora Trader"],
                        cwd=str(self._project_root),
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                self._log.info("Git repository initialised and first commit created")
            except subprocess.TimeoutExpired:
                self._log.warning("Git repo initialisation timed out")
            except subprocess.CalledProcessError as exc:
                self._log.warning(f"Git repo init failed (non-fatal): {exc.stderr}")
            except FileNotFoundError:
                self._log.warning("Git not found — running without version control")

    def _git_commit_and_tag(
        self, strategy_name: str, params_dict: Dict[str, Any], tag: str
    ) -> str:
        """Commit strategy config changes and create an annotated tag.

        Returns the full commit hash, or empty string on failure.
        """
        if not self._auto_commit:
            return ""

        # Build commit message
        params_summary = ", ".join(f"{k}={v}" for k, v in params_dict.items())
        commit_msg = f"strategy({strategy_name}): {tag}\n\n{params_summary}"
        tag_msg = f"Strategy {strategy_name} version {tag}"

        try:
            # Stage all changed files (strategy configs)
            subprocess.run(
                ["git", "add", "-A"],
                cwd=str(self._project_root),
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )

            # Check if there's anything to commit
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(self._project_root),
                capture_output=True,
                text=True,
                timeout=30,
            )
            if not status.stdout.strip():
                # No changes — maybe just tag the current HEAD
                self._log.debug("No file changes to commit")
                # Still tag (or find existing tag to reuse)
                try:
                    # Check if tag already exists
                    tag_check = subprocess.run(
                        ["git", "tag", "-l", tag],
                        cwd=str(self._project_root),
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if tag_check.stdout.strip() == tag:
                        # Tag exists, get its commit hash
                        log = subprocess.run(
                            ["git", "rev-parse", "HEAD"],
                            cwd=str(self._project_root),
                            capture_output=True,
                            text=True,
                            timeout=30,
                        )
                        return log.stdout.strip()
                    else:
                        # Create tag on current commit
                        subprocess.run(
                            ["git", "tag", "-a", tag, "-m", tag_msg],
                            cwd=str(self._project_root),
                            capture_output=True,
                            text=True,
                            check=True,
                            timeout=30,
                        )
                except subprocess.CalledProcessError:
                    pass

                log = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=str(self._project_root),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                return log.stdout.strip()

            # Commit
            subprocess.run(
                ["git", "commit", "-m", commit_msg],
                cwd=str(self._project_root),
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )

            # Tag
            if self._auto_tag:
                subprocess.run(
                    ["git", "tag", "-a", tag, "-m", tag_msg],
                    cwd=str(self._project_root),
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30,
                )

            # Get commit hash
            log = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(self._project_root),
                capture_output=True,
                text=True,
                timeout=30,
            )
            return log.stdout.strip()

        except subprocess.TimeoutExpired:
            self._log.warning("Git commit/tag timed out")
            return ""
        except subprocess.CalledProcessError as exc:
            self._log.warning(f"Git commit/tag failed: {exc.stderr}")
            return ""
        except FileNotFoundError:
            self._log.warning("Git not found — skipping commit/tag")
            return ""

    # ------------------------------------------------------------------
    # Version metadata helpers
    # ------------------------------------------------------------------

    def _hash_params(self, params_dict: Dict[str, Any]) -> str:
        """Create a deterministic SHA-256 hash of the parameter dict."""
        serialized = json.dumps(params_dict, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _bump_version(self, strategy_name: str) -> str:
        """Bump the patch version for the given strategy.

        Determines the next version by looking at existing tags.
        Pattern: v<major>.<minor>.<patch> (e.g. v1.0.0, v1.0.1).

        If no previous versions exist, starts at v1.0.0.
        """
        latest = self.get_latest_version(strategy_name)
        if latest is None:
            return f"{self._tag_prefix}1.0.0"

        # Parse existing tag
        try:
            version_str = latest.lstrip(self._tag_prefix)
            parts = version_str.split(".")
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
            patch += 1
            return f"{self._tag_prefix}{major}.{minor}.{patch}"
        except (ValueError, IndexError):
            # If parsing fails, start fresh
            return f"{self._tag_prefix}1.0.0"

    def _get_version_by_hash(self, strategy_name: str, params_hash: str) -> Optional[str]:
        """Check if an identical parameter set was already versioned."""
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "SELECT version_tag FROM strategy_versions "
                "WHERE strategy_name = ? AND params_hash = ? "
                "ORDER BY rowid DESC LIMIT 1",
                (strategy_name, params_hash),
            )
            row = cursor.fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # SQLite
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create the version tracking table if it doesn't exist."""
        conn = self._get_conn()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS strategy_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version_tag TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    params_hash TEXT NOT NULL,
                    deployed_at TEXT NOT NULL,
                    git_commit_hash TEXT DEFAULT '',
                    description TEXT DEFAULT '',
                    parameters TEXT DEFAULT '{}'
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_strategy_versions_tag "
                "ON strategy_versions(version_tag)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_strategy_versions_strategy "
                "ON strategy_versions(strategy_name)"
            )
            conn.commit()
        except sqlite3.Error as exc:
            self._log.error(f"Failed to initialise version DB: {exc}")
            raise
        finally:
            conn.close()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a new SQLite connection (thread-safe for synchronous ops)."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _insert_version(
        self,
        strategy_name: str,
        tag: str,
        params_hash: str,
        params_dict: Dict[str, Any],
        git_commit_hash: str,
    ) -> None:
        """Insert a new version record into the database."""
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT INTO strategy_versions "
                "(version_tag, strategy_name, params_hash, deployed_at, "
                " git_commit_hash, parameters) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    tag,
                    strategy_name,
                    params_hash,
                    datetime.now(timezone.utc).isoformat(),
                    git_commit_hash,
                    json.dumps(params_dict, default=str),
                ),
            )
            conn.commit()
        except sqlite3.Error as exc:
            self._log.error(f"Failed to insert version {tag}: {exc}")
            raise
        finally:
            conn.close()
