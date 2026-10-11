"""Minimal .env reader so scripts work without extra dependencies, and the Postgres
connection used by the loaders."""

from __future__ import annotations

import os
from pathlib import Path


def read_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    for line in Path(path).read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def connect():
    """psycopg connection to SUPABASE_DB_URL (environment or .env)."""
    read_dotenv()
    import psycopg  # optional dependency: pip install -e ".[db]"

    db_url = os.environ.get("SUPABASE_DB_URL")
    if not db_url:
        raise SystemExit("SUPABASE_DB_URL is not set (see .env.example)")
    return psycopg.connect(db_url)
