# data_sources.py
# Handles all data loading from ESPN (games, odds, schedules) and CSV (legacy).

import csv
import os
import json
import re
from io import StringIO
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urljoin
import logging

import requests
from requests.exceptions import RequestException


CACHE_DIR = Path(__file__).parent / "data" / "cache"
SCOREBOARD_TTL_SECONDS = 300  # 5 minutes for today's scoreboard/odds
PAST_DAY_CACHE_TTL = None  # unlimited for completed games
SCHEDULE_TTL_SECONDS = 600  # 10 minutes for schedules/odds snapshots
ROSTER_TTL_SECONDS = 1800  # 30 minutes
PLAYER_STATS_TTL_SECONDS = 1800  # 30 minutes
MONEYPUCK_TTL_SECONDS = 21600  # 6 hours
MONEYPUCK_SEASON = os.getenv("MONEYPUCK_SEASON")  # e.g., "2025"
NHL_TEAM_STATS_TTL_SECONDS = 21600  # 6 hours
APISPORTS_API_KEY = os.getenv("API_SPORTS_KEY")
APISPORTS_HOCKEY_LEAGUE = os.getenv("APISPORTS_HOCKEY_LEAGUE", "57")  # NHL league id in API-Sports
APISPORTS_HOCKEY_SEASON = os.getenv("APISPORTS_HOCKEY_SEASON")  # e.g., "2024" for 2024-2025 season
APISPORTS_TTL_SECONDS = 600
INJURY_URL = os.getenv(
    "INJURY_URL",
    "https://www.rotowire.com/hockey/tables/injury-report.php?team=ALL&pos=ALL",
)
INJURY_TTL_SECONDS = 4 * 3600  # 4 hours

logger = logging.getLogger("data_sources")


def _normalize_name_key(name: str) -> str:
    """Normalize player name for matching across sources."""
    if not name:
        return ""
    key = name.lower()
    key = re.sub(r"[.\-']", "", key)
    key = re.sub(r"\s+", " ", key).strip()
    return key

# Normalize ESPN abbreviations to our 3-letter set
ALIAS_MAP = {
    "ARI": "UTA",
    "NJ": "NJD",
    "LA": "LAK",
    "TB": "TBL",
    "SJ": "SJS",
    "LV": "VGK",
}

TEAM_NAME_TO_CODE = {
    "ANAHEIM DUCKS": "ANA",
    "ARIZONA COYOTES": "UTA",
    "UTAH HOCKEY CLUB": "UTA",
    "BOSTON BRUINS": "BOS",
    "BUFFALO SABRES": "BUF",
    "CAROLINA HURRICANES": "CAR",
    "COLUMBUS BLUE JACKETS": "CBJ",
    "CALGARY FLAMES": "CGY",
    "CHICAGO BLACKHAWKS": "CHI",
    "COLORADO AVALANCHE": "COL",
    "DALLAS STARS": "DAL",
    "DETROIT RED WINGS": "DET",
    "EDMONTON OILERS": "EDM",
    "FLORIDA PANTHERS": "FLA",
    "LOS ANGELES KINGS": "LAK",
    "MINNESOTA WILD": "MIN",
    "MONTREAL CANADIENS": "MTL",
    "NEW JERSEY DEVILS": "NJD",
    "NASHVILLE PREDATORS": "NSH",
    "NEW YORK ISLANDERS": "NYI",
    "NEW YORK RANGERS": "NYR",
    "OTTAWA SENATORS": "OTT",
    "PHILADELPHIA FLYERS": "PHI",
    "PITTSBURGH PENGUINS": "PIT",
    "SEATTLE KRAKEN": "SEA",
    "SAN JOSE SHARKS": "SJS",
    "ST. LOUIS BLUES": "STL",
    "TAMPA BAY LIGHTNING": "TBL",
    "TORONTO MAPLE LEAFS": "TOR",
    "VANCOUVER CANUCKS": "VAN",
    "VEGAS GOLDEN KNIGHTS": "VGK",
    "WASHINGTON CAPITALS": "WSH",
    "WINNIPEG JETS": "WPG",
}


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


def _read_text(url: str, timeout: int = 10):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; injury-fetch/1.0)"}
        resp = requests.get(url, timeout=timeout, headers=headers)
        resp.raise_for_status()
        return resp.text
    except RequestException as e:
        try:
            status = getattr(e.response, "status_code", None)
        except Exception:
            status = None
        logger.warning("Injury fetch failed (status=%s): %s", status, e)
        return None


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


def _normalize_abbr(code: str) -> str:
    code = (code or "").upper()
    return ALIAS_MAP.get(code, code)


def _fetch_json_cached(url, cache_name: str, ttl_seconds: Optional[int] = None, force_refresh: bool = False):
    cache_path = CACHE_DIR / cache_name
    if not force_refresh:
        cached = _read_cache(cache_path, ttl_seconds=ttl_seconds)
        if cached is not None:
            return cached

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        _write_cache(cache_path, data)
        return data
    except RequestException:
        return _read_cache(cache_path, ttl_seconds=None)  # stale fallback
    except Exception:
        return None


def _apisports_get(path: str, params: dict, cache_name: str, ttl_seconds: Optional[int] = None, force_refresh: bool = False):
    """
    Call API-Sports Hockey endpoint with caching. Returns JSON or None.
    """
    if not APISPORTS_API_KEY:
        return None

    base_url = "https://v1.hockey.api-sports.io"
    cache_path = CACHE_DIR / cache_name

    if not force_refresh:
        cached = _read_cache(cache_path, ttl_seconds=ttl_seconds)
        if cached is not None:
            return cached

    headers = {"x-apisports-key": APISPORTS_API_KEY}
    url = f"{base_url}{path}"
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        _write_cache(cache_path, data)
        return data
    except RequestException:
        return _read_cache(cache_path, ttl_seconds=None)  # stale fallback
    except Exception:
        return None


def _fetch_csv_cached(url: str, cache_name: str, ttl_seconds: Optional[int] = None, force_refresh: bool = False) -> Optional[str]:
    cache_path = CACHE_DIR / cache_name
    if not force_refresh:
        cached = _read_cache(cache_path, ttl_seconds=ttl_seconds)
        if isinstance(cached, str):
            return cached

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        text = response.text
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text, encoding="utf-8")
        return text
    except RequestException:
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8")
        return None
    except Exception:
        return None


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
        home_team = _normalize_abbr(home_team)
        away_team = _normalize_abbr(away_team)
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
    # Try API-Sports first if key is present; fallback to ESPN
    if APISPORTS_API_KEY:
        schedule_api = load_schedule_for_date_apisports(date_str, force_refresh=force_refresh)
        if schedule_api:
            return schedule_api

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
            home_team = _normalize_abbr(home["team"]["abbreviation"])
            away_team = _normalize_abbr(away["team"]["abbreviation"])
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
    # Try API-Sports first if configured
    api_odds = load_current_odds_for_matchup_apisports(home_team, away_team, force_refresh=force_refresh)
    if api_odds:
        return api_odds

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
            home_abbr = _normalize_abbr(home["team"]["abbreviation"])
            away_abbr = _normalize_abbr(away["team"]["abbreviation"])
        except KeyError:
            continue

        # Match on abbreviations (normalized)
        if home_abbr != _normalize_abbr(home_team) or away_abbr != _normalize_abbr(away_team):
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


def read_pick_cache(pick_date: str) -> Optional[dict]:
    """Read a cached pick for a given date string (YYYY-MM-DD)."""
    cache_path = CACHE_DIR / f"pick_{pick_date}.json"
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_pick_cache(pick_date: str, data: dict) -> None:
    """Write pick cache for a given date string."""
    cache_path = CACHE_DIR / f"pick_{pick_date}.json"
    _write_cache(cache_path, data)


def load_team_roster(team_code: str, force_refresh: bool = False) -> Optional[dict]:
    """Load roster JSON for a team from ESPN."""
    team_id = TEAM_CODE_TO_ID.get(team_code)
    if team_id is None:
        return None
    url = f"https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/teams/{team_id}/roster"
    return _fetch_json_cached(
        url,
        cache_name=f"roster_{team_code}.json",
        ttl_seconds=ROSTER_TTL_SECONDS,
        force_refresh=force_refresh,
    )


def load_goalie_stats(athlete_id: str, force_refresh: bool = False) -> Optional[dict]:
    """Load goalie stats JSON for an athlete from ESPN."""
    url = f"https://site.web.api.espn.com/apis/common/v3/sports/hockey/nhl/athletes/{athlete_id}/stats?region=us&lang=en"
    return _fetch_json_cached(
        url,
        cache_name=f"goalie_{athlete_id}.json",
        ttl_seconds=PLAYER_STATS_TTL_SECONDS,
        force_refresh=force_refresh,
    )


def _parse_goalie_stat_value(stat_obj, key_names):
    for key in key_names:
        val = stat_obj.get(key)
        if val is None:
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return None


def _goalie_stats_from_payload(payload) -> dict:
    """Extract simple goalie stats (save_pct, games_started) from ESPN stats payload."""
    save_pct = None
    games_started = None
    games_played = None
    gaa = None

    try:
        stats_list = payload.get("stats", [])
    except Exception:
        stats_list = []

    for block in stats_list:
        try:
            splits = block.get("splits", [])
        except Exception:
            continue
        for split in splits:
            stat = split.get("stat", {})
            save_pct = save_pct or _parse_goalie_stat_value(stat, ["savePct", "savePctAvg"])
            games_started = games_started or _parse_goalie_stat_value(stat, ["gamesStarted", "gamesStartedAvg"])
            games_played = games_played or _parse_goalie_stat_value(stat, ["gamesPlayed"])
            gaa = gaa or _parse_goalie_stat_value(stat, ["goalsAgainstAverage"])
        if save_pct or games_started:
            break

    return {
        "save_pct": save_pct,
        "games_started": games_started,
        "games_played": games_played,
        "gaa": gaa,
    }


def get_probable_goalie(team_code: str, force_refresh: bool = False) -> Optional[dict]:
    """
    Heuristic probable goalie: pick the goalie with most games started; break ties by save_pct.
    """
    roster = load_team_roster(team_code, force_refresh=force_refresh)
    if not roster:
        return None

    try:
        athletes = roster.get("athletes", [])
    except Exception:
        return None

    goalies = []
    for group in athletes:
        try:
            if group.get("position", {}).get("abbreviation") != "G":
                continue
        except Exception:
            continue
        for g in group.get("items", []):
            try:
                athlete_id = str(g["id"])
                full_name = g.get("fullName") or g.get("displayName")
            except Exception:
                continue
            stats_payload = load_goalie_stats(athlete_id, force_refresh=force_refresh)
            stats = _goalie_stats_from_payload(stats_payload) if stats_payload else {}
            goalies.append(
                {
                    "id": athlete_id,
                    "name": full_name,
                    "stats": stats,
                }
            )

    if not goalies:
        return None

    def sort_key(g):
        gs = g["stats"].get("games_started") or 0
        sv = g["stats"].get("save_pct") or 0
        return (gs, sv)

    goalies.sort(key=sort_key, reverse=True)
    return goalies[0]


def _current_season_slug() -> str:
    """Return MoneyPuck season segment (e.g., 2025)."""
    if MONEYPUCK_SEASON:
        return MONEYPUCK_SEASON
    today = datetime.today()
    # MoneyPuck season folders use the END year (e.g., 2025 for 2024-2025).
    season_year = today.year if today.month >= 8 else today.year
    return str(season_year)


def _apisports_hockey_season() -> str:
    """
    Return API-Sports hockey season param.
    API-Sports expects a single year string like '2024' for the 2024-2025 season.
    """
    if APISPORTS_HOCKEY_SEASON:
        return APISPORTS_HOCKEY_SEASON
    today = datetime.today()
    return str(today.year if today.month >= 8 else today.year - 1)


def load_moneypuck_team_stats(force_refresh: bool = False) -> dict:
    """
    Load MoneyPuck team season summary (xG/HDCF/PP/PK) from CSV. Returns a dict keyed by team code.
    """
    season = _current_season_slug()
    url = f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{season}/regular/teams.csv"
    text = _fetch_csv_cached(
        url,
        cache_name=f"moneypuck_team_{season}.csv",
        ttl_seconds=MONEYPUCK_TTL_SECONDS,
        force_refresh=force_refresh,
    )
    if not text:
        return {}

    def _get_first(row, keys):
        for k in keys:
            if k in row and row[k] not in ("", None):
                try:
                    return float(row[k])
                except (TypeError, ValueError):
                    continue
        return None

    def _get_pct(row, keys):
        val = _get_first(row, keys)
        if val is None:
            return None
        return val * 100 if val <= 1 else val

    def _ratio_pct(row, num_key, den_key):
        try:
            num = float(row.get(num_key, "") or 0)
            den = float(row.get(den_key, "") or 0)
        except ValueError:
            return None
        total = num + den
        return (num / total * 100) if total > 0 else None

    teams = {}
    reader = csv.DictReader(StringIO(text))
    for row in reader:
        # Keep only the all-situations team row
        if (row.get("position") != "Team Level") or (row.get("situation", "").lower() != "other"):
            continue
        abbr = _normalize_abbr(row.get("team") or row.get("teamAbbrev") or row.get("team_abbrev"))
        if not abbr:
            continue
        teams[abbr] = {
            "xgf_pct": _get_pct(row, ["xGoalsPercentage", "xGoalsPct"]),
            "xgf_per60": _get_first(row, ["xGoalsForPer60", "xGoalsPer60"]),
            "xga_per60": _get_first(row, ["xGoalsAgainstPer60"]),
            # high-danger share: for / (for + against)
            "hdf_pct": _ratio_pct(row, "highDangerShotsFor", "highDangerShotsAgainst"),
            "pp": _get_pct(row, ["powerPlayPercentage", "ppPct"]),
            "pk": _get_pct(row, ["penaltyKillPercentage", "pkPct"]),
        }

    return teams


def get_team_adv_stats(team_code: str, force_refresh: bool = False) -> Optional[dict]:
    """Return advanced stats dict for team if available."""
    stats = load_moneypuck_team_stats(force_refresh=force_refresh)
    return stats.get(_normalize_abbr(team_code))


def load_nhl_team_stats(force_refresh: bool = False) -> dict:
    """
    Load team stats from the NHL public API (power play, penalty kill, faceoffs, shots).
    Returns a dict keyed by team abbreviation.
    """
    cache_name = "nhl_team_stats.json"
    cache_path = CACHE_DIR / cache_name
    if not force_refresh:
        cached = _read_cache(cache_path, ttl_seconds=NHL_TEAM_STATS_TTL_SECONDS)
        if cached:
            return cached

    url = "https://statsapi.web.nhl.com/api/v1/teams?expand=team.stats"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except RequestException:
        return _read_cache(cache_path, ttl_seconds=None) or {}
    except Exception:
        return {}

    result = {}
    for team in data.get("teams", []):
        try:
            abbr = _normalize_abbr(team.get("abbreviation"))
            stats = team.get("teamStats", [])[0]["splits"][0]["stat"]
        except Exception:
            continue
        if not abbr or not stats:
            continue
        result[abbr] = {
            "pp_pct": _safe_float(stats.get("powerPlayPercentage")),
            "pk_pct": _safe_float(stats.get("penaltyKillPercentage")),
            "fo_pct": _safe_float(stats.get("faceOffWinPercentage")),
            "shots_for": _safe_float(stats.get("shotsPerGame")),
            "shots_against": _safe_float(stats.get("shotsAllowed")),
            "wins": _safe_int(stats.get("wins")),
            "losses": _safe_int(stats.get("losses")),
        }

    _write_cache(cache_path, result)
    return result


def _is_important_injury(entry: dict, player_stats: Optional[dict] = None) -> bool:
    """
    Determine if an injured player is a key contributor based on usage/performance.
    Option A thresholds:
    - Skater: games_played >= 5 and toi_per_game > 12
    - Goalie: starts (or games_played) >= 5 and save_pct >= 0.895 (or save_pct missing but games threshold met)
    Missing stats => False.
    """
    if not player_stats:
        return False
    player = entry.get("player")
    if not player:
        return False
    stats = player_stats.get(_normalize_name_key(player))
    if not stats:
        return False

    is_goalie = bool(stats.get("is_goalie")) or (entry.get("pos") == "G")
    games = stats.get("games_played") or stats.get("starts")
    if games is None:
        return False

    # Option A thresholds
    if is_goalie:
        save_pct = stats.get("save_pct")
        if save_pct is None:
            return games >= 5
        return games >= 5 and save_pct >= 0.895
    toi = stats.get("toi_per_game")
    return games >= 5 and toi is not None and toi > 12


def _normalize_injury_entry(entry: dict, player_stats: Optional[dict] = None) -> Optional[dict]:
    try:
        team = entry.get("team")
        player = entry.get("player")
        if not team or not player:
            return None
        important = _is_important_injury(entry, player_stats=player_stats)
        return {
            "team": team,
            "player": player,
            "position": entry.get("pos") or entry.get("position"),
            "injury": entry.get("injury"),
            "status": entry.get("status"),
            "return": entry.get("estReturn"),
            "note": entry.get("note") or entry.get("notes"),
            "updated": entry.get("updateDate") or entry.get("date"),
            "important": important,
        }
    except Exception:
        return None


def load_injuries_rotowire(force_refresh: bool = False, player_stats: Optional[dict] = None):
    """
    Fetch rotowire injury report JSON and cache it.
    """
    cache_path = CACHE_DIR / "injuries_rotowire.json"
    if not force_refresh:
        cached = _read_cache(cache_path, ttl_seconds=INJURY_TTL_SECONDS)
        if cached is not None:
            return cached

    text = _read_text(INJURY_URL)
    if text is None:
        cached = _read_cache(cache_path, ttl_seconds=None)
        if cached is None:
            logger.error("Injury fetch failed and no cache available.")
        return cached
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            body = data.get("body") or data.get("items") or []
        elif isinstance(data, list):
            body = data
        else:
            logger.error("Unexpected injury payload type: %s", type(data))
            return _read_cache(cache_path, ttl_seconds=None)
        normalized = []
        for entry in body:
            norm = _normalize_injury_entry(entry, player_stats=player_stats)
            if norm:
                normalized.append(norm)
        if not normalized:
            # do not overwrite cache with empty; fall back to last good
            return _read_cache(cache_path, ttl_seconds=None)
        payload = {
            "source": "rotowire",
            "fetched_at": datetime.utcnow().isoformat() + "Z",
            "count": len(normalized),
            "items": normalized,
        }
        _write_cache(cache_path, payload)
        return payload
    except Exception as e:
        logger.exception("Injury fetch parse failed: %s", e)
        return _read_cache(cache_path, ttl_seconds=None)


def load_moneypuck_player_stats(force_refresh: bool = False) -> dict:
    """
    Load MoneyPuck player season summaries (skaters + goalies) and return a dict keyed by player name.
    Fields: games_played, toi_per_game (minutes), is_goalie, save_pct (goalies only, may be None).
    """
    season = _current_season_slug()
    players = {}

    def _add_player(row, is_goalie=False):
        try:
            name = row.get("name")
            games = row.get("games_played")
            icetime = row.get("icetime")
            try:
                games = float(games) if games not in (None, "") else None
            except (TypeError, ValueError):
                games = None
            try:
                icetime = float(icetime) if icetime not in (None, "") else None
            except (TypeError, ValueError):
                icetime = None
            toi_pg = (icetime / games) if games and icetime else None
            save_pct = None
            if is_goalie:
                sv = row.get("savePercentage")
                if sv not in (None, ""):
                    try:
                        save_pct = float(sv)
                    except (TypeError, ValueError):
                        save_pct = None
            if not name or games is None:
                return
            key = _normalize_name_key(name)
            players[key] = {
                "games_played": games,
                "toi_per_game": toi_pg,
                "is_goalie": is_goalie,
                "save_pct": save_pct,
            }
        except Exception:
            return

    # Skaters
    skater_url = f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{season}/regular/skaters.csv"
    skater_text = _fetch_csv_cached(
        skater_url,
        cache_name=f"moneypuck_players_skaters_{season}.csv",
        ttl_seconds=MONEYPUCK_TTL_SECONDS,
        force_refresh=force_refresh,
    )
    if skater_text:
        reader = csv.DictReader(StringIO(skater_text))
        for row in reader:
            _add_player(row, is_goalie=False)

    # Goalies
    goalie_url = f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{season}/regular/goalies.csv"
    goalie_text = _fetch_csv_cached(
        goalie_url,
        cache_name=f"moneypuck_players_goalies_{season}.csv",
        ttl_seconds=MONEYPUCK_TTL_SECONDS,
        force_refresh=force_refresh,
    )
    if goalie_text:
        reader = csv.DictReader(StringIO(goalie_text))
        for row in reader:
            _add_player(row, is_goalie=True)

    return players


def _safe_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _safe_int(val):
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


# Legacy NHL Stats API bits (network blocked on your side)
TEAM_CODE_TO_ID = {
    "ANA": 25,
    "ARI": 129764,  # Utah/Arizona franchise
    "UTA": 129764,
    "BOS": 1,
    "BUF": 2,
    "CAR": 7,
    "CBJ": 29,
    "CGY": 3,
    "CHI": 4,
    "COL": 17,
    "DAL": 9,
    "DET": 5,
    "EDM": 6,
    "FLA": 26,
    "LAK": 8,  # ESPN uses LA
    "MIN": 30,
    "MTL": 10,
    "NJD": 11,  # ESPN uses NJ
    "NSH": 27,
    "NYI": 12,
    "NYR": 13,
    "OTT": 14,
    "PHI": 15,
    "PIT": 16,
    "SEA": 124292,
    "SJS": 18,  # ESPN uses SJ
    "STL": 19,
    "TBL": 20,  # ESPN uses TB
    "TOR": 21,
    "VAN": 22,
    "VGK": 37,
    "WPG": 28,
    "WSH": 23,
}


def _name_to_code(name: str) -> Optional[str]:
    if not name:
        return None
    cleaned = " ".join(name.upper().replace(".", "").split())
    return TEAM_NAME_TO_CODE.get(cleaned)


def load_schedule_for_date_apisports(date_str: str, force_refresh: bool = False):
    """
    Load schedule for a specific date from API-Sports Hockey if configured.
    Returns same shape as ESPN schedule loader: list of {date, home_team, away_team}.
    """
    if not APISPORTS_API_KEY:
        return []
    season = _apisports_hockey_season()
    params = {
        "league": APISPORTS_HOCKEY_LEAGUE,
        "season": season,
        "date": date_str,
    }
    data = _apisports_get(
        "/games",
        params=params,
        cache_name=f"apisports_games_{date_str}.json",
        ttl_seconds=APISPORTS_TTL_SECONDS,
        force_refresh=force_refresh,
    )
    if not data:
        return []

    games = []
    try:
        responses = data.get("response", [])
    except Exception:
        responses = []
    for g in responses:
        try:
            fixture = g.get("fixture", {})
            game_date = fixture.get("date")
            teams = g.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            home_code = _name_to_code(home.get("name"))
            away_code = _name_to_code(away.get("name"))
            if not home_code or not away_code:
                continue
            games.append(
                {
                    "date": game_date,
                    "home_team": home_code,
                    "away_team": away_code,
                }
            )
        except Exception:
            continue
    return games


def load_current_odds_for_matchup_apisports(home_team: str, away_team: str, date_str: Optional[str] = None, force_refresh: bool = False) -> Optional[dict]:
    """
    Try to fetch odds for a matchup from API-Sports Hockey.
    Returns {home_ml: float, away_ml: float} or None.
    """
    if not APISPORTS_API_KEY:
        return None
    season = _apisports_hockey_season()
    date_param = date_str or datetime.today().strftime("%Y-%m-%d")
    params = {
        "league": APISPORTS_HOCKEY_LEAGUE,
        "season": season,
        "date": date_param,
    }
    data = _apisports_get(
        "/odds",
        params=params,
        cache_name=f"apisports_odds_{date_param}.json",
        ttl_seconds=APISPORTS_TTL_SECONDS,
        force_refresh=force_refresh,
    )
    if not data:
        return None

    try:
        responses = data.get("response", [])
    except Exception:
        responses = []

    home_norm = _normalize_abbr(home_team)
    away_norm = _normalize_abbr(away_team)

    for item in responses:
        try:
            game = item.get("game", {})
            teams = game.get("teams", {})
            home_name = teams.get("home", {}).get("name")
            away_name = teams.get("away", {}).get("name")
            home_code = _name_to_code(home_name)
            away_code = _name_to_code(away_name)
            if home_code != home_norm or away_code != away_norm:
                continue

            bookmakers = item.get("bookmakers", [])
            if not bookmakers:
                continue
            # Use first bookmaker, first moneyline-style bet
            home_ml = None
            away_ml = None
            for bk in bookmakers:
                for bet in bk.get("bets", []):
                    name = (bet.get("name") or "").lower()
                    if "winner" not in name and "moneyline" not in name:
                        continue
                    for v in bet.get("values", []):
                        val_name = (v.get("value") or "").lower()
                        try:
                            price = float(v.get("odd"))
                        except (TypeError, ValueError):
                            continue
                        if val_name in ("home", "1", (home_name or "").lower()):
                            home_ml = price
                        if val_name in ("away", "2", (away_name or "").lower()):
                            away_ml = price
                    if home_ml is not None and away_ml is not None:
                        break
                if home_ml is not None and away_ml is not None:
                    break

            if home_ml is None or away_ml is None:
                continue
            return {"home_ml": home_ml, "away_ml": away_ml}
        except Exception:
            continue

    return None

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
