"""
Ingest SBR-style NFL odds CSVs into a normalized per-game file.

Example usage:
    python scripts/ingest_nfl_odds.py \
        --raw data/odds/raw/nfl/nfl_2023_raw.csv \
        --season 2023 \
        --out data/odds/clean/nfl/nfl_odds_2023.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from typing import Dict, Optional


# Common variations -> ESPN team codes (sanitized: lowercase, no spaces/dots)
TEAM_MAP: Dict[str, str] = {
    "arizonacardinals": "ARI",
    "arizona": "ARI",
    "cardinals": "ARI",
    "atlantafalcons": "ATL",
    "atlanta": "ATL",
    "falcons": "ATL",
    "baltimoreravens": "BAL",
    "baltimore": "BAL",
    "ravens": "BAL",
    "buffalobills": "BUF",
    "buffalo": "BUF",
    "bills": "BUF",
    "carolinapanthers": "CAR",
    "carolina": "CAR",
    "panthers": "CAR",
    "chicagobears": "CHI",
    "chicago": "CHI",
    "bears": "CHI",
    "cincinnatibengals": "CIN",
    "cincinnati": "CIN",
    "bengals": "CIN",
    "clevelandbrowns": "CLE",
    "cleveland": "CLE",
    "browns": "CLE",
    "dallascowboys": "DAL",
    "dallas": "DAL",
    "cowboys": "DAL",
    "denverbroncos": "DEN",
    "denver": "DEN",
    "broncos": "DEN",
    "detroitlions": "DET",
    "detroit": "DET",
    "lions": "DET",
    "greenbaypackers": "GB",
    "greenbay": "GB",
    "packers": "GB",
    "houstontexans": "HOU",
    "houston": "HOU",
    "texans": "HOU",
    "indianapoliscolts": "IND",
    "indianapolis": "IND",
    "colts": "IND",
    "jacksonvillejaguars": "JAX",
    "jacksonville": "JAX",
    "jaguars": "JAX",
    "kansascitychiefs": "KC",
    "kansascity": "KC",
    "kansas": "KC",
    "chiefs": "KC",
    "kcchiefs": "KC",
    "lasvegasraiders": "LV",
    "lasvegas": "LV",
    "raiders": "LV",
    "lvraiders": "LV",
    "oaklandraiders": "LV",
    "oakland": "LV",
    "losangeleschargers": "LAC",
    "lachargers": "LAC",
    "chargers": "LAC",
    "sandiegochargers": "LAC",
    "sandiego": "LAC",
    "losangelesrams": "LAR",
    "larams": "LAR",
    "rams": "LAR",
    "stlouisrams": "LAR",
    "stlouis": "LAR",
    "miamidolphins": "MIA",
    "miami": "MIA",
    "dolphins": "MIA",
    "minnesotavikings": "MIN",
    "minnesota": "MIN",
    "vikings": "MIN",
    "newenglandpatriots": "NE",
    "newengland": "NE",
    "patriots": "NE",
    "neworleanssaints": "NO",
    "neworleans": "NO",
    "saints": "NO",
    "newyorkgiants": "NYG",
    "nygiants": "NYG",
    "giants": "NYG",
    "newyorkjets": "NYJ",
    "nyjets": "NYJ",
    "jets": "NYJ",
    "philadelphiaeagles": "PHI",
    "philadelphia": "PHI",
    "eagles": "PHI",
    "pittsburghsteelers": "PIT",
    "pittsburgh": "PIT",
    "steelers": "PIT",
    "sanfrancisco49ers": "SF",
    "sanfrancisco": "SF",
    "49ers": "SF",
    "seattleseahawks": "SEA",
    "seattle": "SEA",
    "seahawks": "SEA",
    "tampabaybuccaneers": "TB",
    "tampabay": "TB",
    "tampa": "TB",
    "buccaneers": "TB",
    "tennesseetitans": "TEN",
    "tennessee": "TEN",
    "titans": "TEN",
    "washingtoncommanders": "WAS",
    "washington": "WAS",
    "washingtom": "WAS",
    "commanders": "WAS",
    "washingtonfootballteam": "WAS",
    "washingtonredskins": "WAS",
    "redskins": "WAS",
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
    if value is None:
        return None
    value = str(value).strip().lower()
    if value == "pk" or value == "":
        return 0.0
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
            if len(row) >= 16:
                # Standard SBR format with spread/total columns
                entry[f"{side_prefix}_score"] = int(row[7]) if row[7] else None
                entry[f"{side_prefix}_ml_open"] = parse_float(row[8])
                entry[f"{side_prefix}_ml_close"] = parse_float(row[9])
                entry[f"{side_prefix}_spread"] = parse_float(row[10])
                entry[f"{side_prefix}_spread_price"] = parse_float(row[11])

                if side_prefix == "away":
                    entry["total_open"] = parse_float(row[12])
                    entry["total_open_price"] = parse_float(row[13])
                    entry["total_close"] = parse_float(row[14])
                    entry["total_close_price"] = parse_float(row[15])
            elif len(row) >= 13:
                # Slim format (SBR NFL ML/spread/total collapsed)
                entry[f"{side_prefix}_score"] = int(row[8]) if row[8] else None
                entry[f"{side_prefix}_ml_open"] = None
                entry[f"{side_prefix}_ml_close"] = parse_float(row[11])
                # Store raw columns; decide spread vs total after both rows processed
                entry[f"{side_prefix}_col9"] = parse_float(row[9])
                entry[f"{side_prefix}_col10"] = parse_float(row[10])
            else:
                raise ValueError(f"Unexpected row length {len(row)} in {row}")

    # Determine spread vs total based on favorite (lower ML)
    for _, data in games.items():
        home_ml = data.get("home_ml_close")
        away_ml = data.get("away_ml_close")
        away_col9 = data.get("away_col9")
        away_col10 = data.get("away_col10")
        home_col9 = data.get("home_col9")
        home_col10 = data.get("home_col10")

        if home_ml is None or away_ml is None:
            continue

        if home_ml < away_ml:
            # Home favorite: home row is spread, away row is total
            data["home_spread"] = -home_col10 if home_col10 is not None else None
            data["home_spread_open"] = -home_col9 if home_col9 is not None else None
            data["away_spread"] = home_col10
            data["away_spread_open"] = home_col9
            data["total_open"] = away_col9
            data["total_close"] = away_col10
        else:
            # Away favorite: away row is spread, home row is total
            data["away_spread"] = -away_col10 if away_col10 is not None else None
            data["away_spread_open"] = -away_col9 if away_col9 is not None else None
            data["home_spread"] = away_col10
            data["home_spread_open"] = away_col9
            data["total_open"] = home_col9
            data["total_close"] = home_col10

        data["home_spread_price"] = -110.0
        data["away_spread_price"] = -110.0
        data["total_open_price"] = None
        data["total_close_price"] = None

        for k in ("home_col9", "home_col10", "away_col9", "away_col10"):
            data.pop(k, None)

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
        "home_spread",
        "home_spread_price",
        "home_spread_open",
        "away_spread",
        "away_spread_open",
        "away_spread_price",
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
    parser = argparse.ArgumentParser(description="Normalize raw SBR NFL odds CSVs.")
    parser.add_argument("--raw", required=True, help="Path to raw CSV (data/odds/raw/nfl/*.csv)")
    parser.add_argument("--out", required=True, help="Path to write normalized CSV")
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="Starting calendar year for the season (e.g., 2023 for 2023 season start)",
    )
    args = parser.parse_args()
    ingest(args.raw, args.out, args.season)
