"""Utility helpers for loading cleaned odds data files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple


OddsKey = Tuple[str, str, str]


def load_clean_odds(path: str) -> Dict[OddsKey, dict]:
    """Load a normalized odds CSV (date/home/away columns)."""
    odds_map: Dict[OddsKey, dict] = {}
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Odds file not found: {path}")

    with file_path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"date", "home_team", "away_team"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Odds file {path} missing columns {required}")

        for row in reader:
            key: OddsKey = (
                row["date"].strip(),
                row["home_team"].strip(),
                row["away_team"].strip(),
            )
            odds_map[key] = row

    return odds_map
