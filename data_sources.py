# data_sources.py
# Handles all data loading from ESPN (games, odds, schedules) and CSV (legacy).

import csv
import json
from pathlib import Path
from datetime import datetime, timedelta

import requests
from requests.exceptions import RequestException


CACHE_DIR = Path(__file__).parent / "data" / "cache"
SCOREBOARD_TTL_SECONDS = 300  # 5 minutes for today's scoreboard/odds
PAST_DAY_CACHE_TTL = None  # unlimited for completed games
SCHEDULE_TTL_SECONDS = 600  # 10 minutes for schedules/odds snapshots


def _read_cache(cache_path, ttl_seconds=None):
    """
    Return cached JSON if it exists and is within TTL.
    """
    if not cache_path.exists():
        return None

    if ttl_seconds is not None:
        age = datetime.now().timestamp() - cache_path.stat().st_mtime
        if age > ttl_seconds:
            return None

    try:
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_cache(cache_path, data):
    """
    Write JSON to cache, ensuring directory exists.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def _fetch_scoreboard_json(
    date_param=None, ttl_seconds=SCOREBOARD_TTL_SECONDS, force_refresh=False
):
    """
    Fetch ESPN scoreboard JSON once per date, with a short-lived cache.
    """
    suffix = date_param or "today"
    cache_path = CACHE_DIR / f"scoreboard_{suffix}.json"

    if not force_refresh:
        cached = _read_cache(cache_path, ttl_seconds=ttl_seconds)
        if cached is not None:
            return cached

    base = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
    url = f"{base}?dates={date_param}" if date_param else base

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except RequestException as e:
        print("\n[API ERROR] Could not reach ESPN NHL scoreboard.")
        print("Details:", e)
        return None

    data = response.json()
    _write_cache(cache_path, data)
    return data


def load_games_from_csv(csv_path):
    """
    Load games from a CSV file into a list of dictionaries.
    (Not used right now, but kept for future use.)
    """
    csv_path = Path(csv_path)
    games = []

    with csv_path.open(mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            games.append(row)

    return games


def _parse_espn_events(events):
    """
    Internal helper to convert ESPN 'events' list into our standard game dicts.

    Each returned dict looks like:
    {
        'date': 'YYYY-MM-DD',
        'home_team': 'XXX',
        'away_team': 'YYY',
        'home_goals': '3',
        'away_goals': '2'
    }

    NOTE: This ONLY includes **completed** games (for historical stats).
    """
    games = []

    for ev in events:
        game_date = ev.get("date", "")[:10]  # e.g. '2025-11-18'
        competitions = ev.get("competitions", [])
        if not competitions:
            continue

        comp = competitions[0]
        status = comp.get("status", {})

        # Only include completed games for historical stats
        completed = status.get("type", {}).get("completed", False)
        if not completed:
            continue

        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue

        # Find home and away teams
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)

        if not home or not away:
            continue

        try:
            home_team = home["team"]["abbreviation"]
            away_team = away["team"]["abbreviation"]
            home_score = str(home.get("score", "0"))
            away_score = str(away.get("score", "0"))
        except KeyError:
            continue  # skip weirdly formatted games

        games.append(
            {
                "date": game_date,
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": home_score,
                "away_goals": away_score,
            }
        )

    return games


def load_games_from_espn_scoreboard(force_refresh=False):
    """
    Load today's completed NHL games from ESPN's public NHL scoreboard API.

    Returns a list of game dicts in the SAME format as our CSV rows.
    """
    data = _fetch_scoreboard_json(
        date_param=None,
        ttl_seconds=SCOREBOARD_TTL_SECONDS,
        force_refresh=force_refresh,
    )
    if data is None:
        return []

    events = data.get("events", [])
    return _parse_espn_events(events)


def load_games_from_espn_date_range(start_date_str, end_date_str, force_refresh=False):
    """
    Load ALL completed NHL games between start_date and end_date (inclusive)
    using ESPN's scoreboard API.

    Dates are strings in 'YYYY-MM-DD' format, e.g. '2024-10-01'.

    This will make one API call per day in the range.
    """
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    except ValueError:
        print("\n[DATE ERROR] Dates must be in YYYY-MM-DD format.")
        return []

    # If user accidentally flips them
    if end_date < start_date:
        start_date, end_date = end_date, start_date

    all_games = []
    current = start_date

    print(f"\nFetching games from {start_date} to {end_date} from ESPN...")

    while current <= end_date:
        date_param = current.strftime("%Y%m%d")  # ESPN expects YYYYMMDD

        data = _fetch_scoreboard_json(
            date_param=date_param,
            ttl_seconds=PAST_DAY_CACHE_TTL,
            force_refresh=force_refresh,
        )

        if data is None:
            print(f"\n[API ERROR] Could not reach ESPN for {current}.")
        else:
            try:
                events = data.get("events", [])
                day_games = _parse_espn_events(events)
                all_games.extend(day_games)
                print(f"  {current}: {len(day_games)} completed games.")
            except Exception as e:
                print(f"\n[PARSE ERROR] Problem reading data for {current}.")
                print("Details:", e)

        current += timedelta(days=1)

    print(f"\nTotal games fetched in range: {len(all_games)}")
    return all_games


def load_schedule_for_date(date_str, force_refresh=False):
    """
    Load the NHL **schedule** for a specific date (future or past).

    This is different from load_games_from_espn_date_range:
    - Here we include games whether or not they've been played
    - We only care about who is playing (home/away abbreviations)

    Returns a list like:
    [
      { "date": "2025-11-20T00:00Z...", "home_team": "BOS", "away_team": "NYR" },
      ...
    ]
    """
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        print("\n[DATE ERROR] Date must be in YYYY-MM-DD format.")
        return []

    date_param = d.strftime("%Y%m%d")
    data = _fetch_scoreboard_json(
        date_param=date_param,
        ttl_seconds=SCHEDULE_TTL_SECONDS,
        force_refresh=force_refresh,
    )
    if data is None:
        return []

    events = data.get("events", [])

    schedule = []

    for ev in events:
        game_date_full = ev.get("date", "")  # full timestamp string
        competitions = ev.get("competitions", [])
        if not competitions:
            continue

        comp = competitions[0]
        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue

        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)

        if not home or not away:
            continue

        try:
            home_team = home["team"]["abbreviation"]
            away_team = away["team"]["abbreviation"]
        except KeyError:
            continue

        schedule.append(
            {
                "date": game_date_full,
                "home_team": home_team,
                "away_team": away_team,
            }
        )

    return schedule


def load_current_odds_for_matchup(home_team, away_team, force_refresh=False):
    """
    Try to fetch ESPN odds for *today's* NHL games for a specific matchup.

    If found and odds info is present, return a dict like:
      { "home_ml": -135, "away_ml": +120 }

    If not found or odds missing, return None.
    """
    data = _fetch_scoreboard_json(
        date_param=None,
        ttl_seconds=SCOREBOARD_TTL_SECONDS,
        force_refresh=force_refresh,
    )
    if data is None:
        return None

    try:
        events = data.get("events", [])
    except Exception as e:
        print("\n[ODDS PARSE ERROR] Could not parse ESPN odds JSON.")
        print("Details:", e)
        return None

    for ev in events:
        competitions = ev.get("competitions", [])
        if not competitions:
            continue

        comp = competitions[0]
        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue

        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)

        if not home or not away:
            continue

        try:
            home_abbr = home["team"]["abbreviation"]
            away_abbr = away["team"]["abbreviation"]
        except KeyError:
            continue

        # Match on abbreviations (e.g. TOR, EDM)
        if home_abbr != home_team or away_abbr != away_team:
            continue

        # Try to read the odds block
        odds_list = comp.get("odds", [])
        if not odds_list:
            return None

        book = odds_list[0]  # main book
        home_odds_info = book.get("homeTeamOdds") or {}
        away_odds_info = book.get("awayTeamOdds") or {}

        home_ml = home_odds_info.get("moneyLine")
        away_ml = away_odds_info.get("moneyLine")

        if home_ml is None or away_ml is None:
            return None

        try:
            return {
                "home_ml": float(home_ml),
                "away_ml": float(away_ml),
            }
        except (TypeError, ValueError):
            return None

    # If we looked through all events and found no matching game
    return None


# Legacy NHL Stats API bits (network blocked on your side)
TEAM_CODE_TO_ID = {
    "ANA": 24,
    "ARI": 53,
    "BOS": 6,
    "BUF": 7,
    "CAR": 12,
    "CBJ": 29,
    "CGY": 20,
    "CHI": 16,
    "COL": 21,
    "DAL": 25,
    "DET": 17,
    "EDM": 22,
    "FLA": 13,
    "LAK": 26,
    "MIN": 30,
    "MTL": 8,
    "NJD": 1,
    "NSH": 18,
    "NYI": 2,
    "NYR": 3,
    "OTT": 9,
    "PHI": 4,
    "PIT": 5,
    "SEA": 55,
    "SJS": 28,
    "STL": 19,
    "TBL": 14,
    "TOR": 10,
    "UTA": 54,  # Utah (former ARI, adjust if needed)
    "VAN": 23,
    "VGK": 54,  # may differ by season
    "WPG": 52,
    "WSH": 15,
}


def load_recent_games_for_team_from_api(team_code, num_games=10):
    """
    OLD: NHL Stats API, blocked on your network.
    Left here only so imports don't break.
    """
    print(
        "\n[INFO] load_recent_games_for_team_from_api uses the NHL Stats API, "
        "which is blocked on your network. Returning empty list."
    )
    return []
