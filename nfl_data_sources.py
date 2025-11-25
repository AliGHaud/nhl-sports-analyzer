# Minimal NFL data loader using ESPN public endpoints.

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests

CACHE_DIR = Path(__file__).parent / "data" / "cache"
SCOREBOARD_TTL_SECONDS = 300

NFL_TEAM_CODES = {
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
    "DET", "GB", "HOU", "IND", "JAX", "KC", "LV", "LAC", "LAR", "MIA",
    "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"
}


def _read_cache(path: Path, ttl_seconds: Optional[int] = None):
    if not path.exists():
        return None
    if ttl_seconds is not None:
        age = datetime.now().timestamp() - path.stat().st_mtime
        if age > ttl_seconds:
            return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_cache(path: Path, data):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def _fetch_scoreboard_json(date_param=None, ttl_seconds=SCOREBOARD_TTL_SECONDS):
    suffix = date_param or "today"
    cache_path = CACHE_DIR / f"nfl_scoreboard_{suffix}.json"
    cached = _read_cache(cache_path, ttl_seconds=ttl_seconds)
    if cached is not None:
        return cached
    base = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    url = f"{base}?dates={date_param}" if date_param else base
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        _write_cache(cache_path, data)
        return data
    except Exception:
        return cached or None


def _parse_events(events, include_unplayed=False):
    games = []
    for ev in events:
        game_date = ev.get("date", "")[:10]
        competitions = ev.get("competitions", [])
        if not competitions:
            continue
        comp = competitions[0]
        status = comp.get("status", {})
        completed = status.get("type", {}).get("completed", False)
        if not completed and not include_unplayed:
            continue
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
            home_score = str(home.get("score", "0"))
            away_score = str(away.get("score", "0"))
        except Exception:
            continue
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


def load_games_from_espn_scoreboard():
    data = _fetch_scoreboard_json(date_param=None, ttl_seconds=SCOREBOARD_TTL_SECONDS)
    if not data:
        return []
    return _parse_events(data.get("events", []))


def load_games_from_espn_date_range(start_date_str, end_date_str):
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    except ValueError:
        return []
    if end_date < start_date:
        start_date, end_date = end_date, start_date
    all_games = []
    current = start_date
    while current <= end_date:
        date_param = current.strftime("%Y%m%d")
        data = _fetch_scoreboard_json(date_param=date_param, ttl_seconds=None)
        if data:
            all_games.extend(_parse_events(data.get("events", [])))
        current += timedelta(days=1)
    return all_games


def load_schedule_for_date(date_str):
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return []
    date_param = d.strftime("%Y%m%d")
    data = _fetch_scoreboard_json(date_param=date_param, ttl_seconds=SCOREBOARD_TTL_SECONDS)
    if not data:
        return []
    return _parse_events(data.get("events", []), include_unplayed=True)


def load_current_odds_for_matchup(home_team, away_team):
    data = _fetch_scoreboard_json(date_param=None, ttl_seconds=SCOREBOARD_TTL_SECONDS)
    if not data:
        return None
    for ev in data.get("events", []):
        comps = ev.get("competitions", [])
        if not comps:
            continue
        comp = comps[0]
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
        except Exception:
            continue
        if home_abbr != home_team or away_abbr != away_team:
            continue
        odds_list = comp.get("odds", [])
        if not odds_list:
            return None
        book = odds_list[0]
        home_odds_info = book.get("homeTeamOdds") or {}
        away_odds_info = book.get("awayTeamOdds") or {}
        home_ml = home_odds_info.get("moneyLine")
        away_ml = away_odds_info.get("moneyLine")
        if home_ml is None or away_ml is None:
            return None
        try:
            return {"home_ml": float(home_ml), "away_ml": float(away_ml)}
        except Exception:
            return None
    return None
