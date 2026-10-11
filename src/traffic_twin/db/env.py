"""Minimal .env reader so scripts work without extra dependencies."""

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
