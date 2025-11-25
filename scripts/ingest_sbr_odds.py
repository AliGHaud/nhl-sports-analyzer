"""
Ingest MoneyPuck/SBR style CSV odds dumps into a normalized per-game file.

Example usage (2021-22 season):
    python scripts/ingest_sbr_odds.py \
        --raw data/odds/raw/sbr/nhl_2021_22_raw.csv \
        --season 2021 \
        --out data/odds/clean/nhl_odds_2021_22.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from typing import Dict, Optional


TEAM_MAP: Dict[str, str] = {
    "anaheim": "ANA",
    "arizona": "ARI",
    "boston": "BOS",
    "buffalo": "BUF",
    "carolina": "CAR",
    "columbus": "CBJ",
    "calgary": "CGY",
    "chicago": "CHI",
    "colorado": "COL",
    "dallas": "DAL",
    "detroit": "DET",
    "edmonton": "EDM",
    "florida": "FLA",
    "losangeles": "LAK",
    "minnesota": "MIN",
    "montreal": "MTL",
    "newjersey": "NJD",
    "nashville": "NSH",
    "nyislanders": "NYI",
    "nyrangers": "NYR",
    "ottawa": "OTT",
    "philadelphia": "PHI",
    "pittsburgh": "PIT",
    "seattleskraken": "SEA",  # legacy typo fallback
    "seattlekraken": "SEA",
    "seattle": "SEA",
    "sanjose": "SJS",
    "stlouis": "STL",
    "tampabay": "TBL",
    "toronto": "TOR",
    "vancouver": "VAN",
    "vegas": "VGK",
    "washington": "WSH",
    "winnipeg": "WPG",
}


def parse_date(code: str, season_start_year: int) -> str:
    s = code.strip().zfill(4)
    month = int(s[:-2])
    day = int(s[-2:])
    # Months July-Dec belong to the season's starting calendar year.
    year = season_start_year if month >= 7 else season_start_year + 1
    return datetime(year, month, day).strftime("%Y-%m-%d")


def sanitize_team(name: str) -> str:
    return name.replace(" ", "").replace(".", "").lower()


def parse_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ingest(raw_path: str, out_path: str, season_start_year: int) -> None:
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw odds file not found: {raw_path}")

    games: Dict[tuple, Dict[str, object]] = {}

    with open(raw_path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # drop short header
        header = next(reader, None)
        if not header or len(header) < 13:
            raise ValueError("Unexpected header format in raw CSV.")

        for row in reader:
            if not row or row[0] == "Date":
                continue

            date_code, rot_str, vh, raw_team = row[:4]
            rot = int(rot_str)
            key_rot = rot if rot % 2 == 1 else rot - 1  # odd rot id per game
            game_date = parse_date(date_code, season_start_year)
            game_key = (game_date, key_rot)

            entry = games.setdefault(
                game_key,
                {
                    "date": game_date,
                    "season_start": season_start_year,
                    "base_rot": key_rot,
                    "home_team": None,
                    "away_team": None,
                },
            )

            team_key = sanitize_team(raw_team)
            team_code = TEAM_MAP.get(team_key)
            if not team_code:
                raise KeyError(f"Unknown team label '{raw_team}' in row {row}")

            side_prefix = "home" if vh.upper() == "H" else "away"
            entry[f"{side_prefix}_team"] = team_code
            entry[f"{side_prefix}_score"] = int(row[7]) if row[7] else None
            entry[f"{side_prefix}_ml_open"] = parse_float(row[8])
            entry[f"{side_prefix}_ml_close"] = parse_float(row[9])
            entry[f"{side_prefix}_puckline"] = parse_float(row[10])
            entry[f"{side_prefix}_puckline_price"] = parse_float(row[11])

            if side_prefix == "away":
                entry["total_open"] = parse_float(row[12])
                entry["total_open_price"] = parse_float(row[13])
                entry["total_close"] = parse_float(row[14])
                entry["total_close_price"] = parse_float(row[15])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = [
        "date",
        "season_start",
        "base_rot",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "home_ml_open",
        "home_ml_close",
        "away_ml_open",
        "away_ml_close",
        "home_puckline",
        "home_puckline_price",
        "away_puckline",
        "away_puckline_price",
        "total_open",
        "total_open_price",
        "total_close",
        "total_close_price",
    ]

    with open(out_path, "w", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        missing = 0
        for _, data in sorted(games.items()):
            if not data.get("home_team") or not data.get("away_team"):
                missing += 1
                continue
            writer.writerow(data)

    print(f"Wrote cleaned odds to {out_path} (games: {len(games) - missing}, skipped: {missing})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize raw SBR odds CSVs.")
    parser.add_argument("--raw", required=True, help="Path to raw CSV (data/odds/raw/sbr/*.csv)")
    parser.add_argument("--out", required=True, help="Path to write normalized CSV")
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="Starting calendar year for the season (e.g., 2021 for 2021-22)",
    )
    args = parser.parse_args()
    ingest(args.raw, args.out, args.season)
