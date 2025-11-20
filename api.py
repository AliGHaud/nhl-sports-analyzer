"""Lightweight FastAPI wrapper around the NHL analyzer logic.

Endpoints:
- GET /health
- GET /nhl/matchup?home=TOR&away=BOS[&start_date=YYYY-MM-DD&end_date=YYYY-MM-DD]
  Fetches recent games (default last 60 days), builds team profiles, runs lean + EV,
  and returns JSON suitable for a front-end.
"""

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from data_sources import (
    load_games_from_espn_date_range,
    load_current_odds_for_matchup,
    load_schedule_for_date,
    TEAM_CODE_TO_ID,
)
from nhl_analyzer import (
    get_team_profile,
    implied_prob_american,
    profit_on_win_for_1_unit,
    model_probs_from_scores,
    confidence_grade,
)

app = FastAPI(title="NHL Betting Analyzer API", version="0.1.0")
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("nhl_analyzer_api")

SPORTS = {
    "nhl": {
        "title": "NHL",
        "teams": sorted(TEAM_CODE_TO_ID.keys()),
    },
}

ALIASES = {
    "ARI": "UTA",  # legacy Arizona to Utah
}


# ---------- Helpers ----------

def _parse_date_or_400(value: Optional[str], fallback: date, label: str) -> date:
    """
    Parse YYYY-MM-DD or raise a 400 with a clear message. Returns fallback if value is None.
    """
    if value is None:
        return fallback
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"{label} must be YYYY-MM-DD (got '{value}').",
        )


def _validate_team_or_400(team: str, label: str) -> str:
    """
    Ensure team is 2-4 letters and in the known NHL code set.
    """
    cleaned = (team or "").strip().upper()
    if not cleaned.isalpha() or not (2 <= len(cleaned) <= 4):
        raise HTTPException(
            status_code=400,
            detail=f"{label} must be a team abbreviation (2-4 letters).",
        )
    cleaned = ALIASES.get(cleaned, cleaned)

    if cleaned not in TEAM_CODE_TO_ID:
        raise HTTPException(
            status_code=400,
            detail=f"{label} '{cleaned}' is not a known NHL team code.",
        )
    return cleaned


def _lean_scores(home_team: str, away_team: str, games) -> Tuple[float, float, dict]:
    """Pure version of the lean logic: returns scores + reasons."""
    home_profile = get_team_profile(home_team, games)
    away_profile = get_team_profile(away_team, games)

    if home_profile is None or away_profile is None:
        raise ValueError("Not enough data to build team profiles.")

    def win_pct(stats):
        g = stats["games"]
        return stats["wins"] / g if g > 0 else 0.0

    home_score = 0.0
    away_score = 0.0
    reasons_home = []
    reasons_away = []

    # Last 10 record
    home_win10 = win_pct(home_profile["last10"])
    away_win10 = win_pct(away_profile["last10"])
    home_score += home_win10 * 2.0
    away_score += away_win10 * 2.0
    if home_win10 > away_win10 + 0.1:
        reasons_home.append("Better last 10 record")
    elif away_win10 > home_win10 + 0.1:
        reasons_away.append("Better last 10 record")

    # Last 5 record
    home_win5 = win_pct(home_profile["last5"])
    away_win5 = win_pct(away_profile["last5"])
    home_score += home_win5 * 1.0
    away_score += away_win5 * 1.0
    if home_win5 > away_win5 + 0.1:
        reasons_home.append("Stronger recent (last 5) form")
    elif away_win5 > home_win5 + 0.1:
        reasons_away.append("Stronger recent (last 5) form")

    # Streaks
    for profile, is_home_flag in ((home_profile, True), (away_profile, False)):
        streak = profile["streak"]
        length = streak["length"]
        stype = streak["type"]
        if length > 0 and stype:
            delta = 0.5 * length if stype == "W" else -0.5 * length
            if is_home_flag:
                home_score += delta
                if stype == "W" and length >= 2:
                    reasons_home.append(f"On a {length}-game winning streak")
                if stype == "L" and length >= 2:
                    reasons_home.append(f"On a {length}-game losing streak")
            else:
                away_score += delta
                if stype == "W" and length >= 2:
                    reasons_away.append(f"On a {length}-game winning streak")
                if stype == "L" and length >= 2:
                    reasons_away.append(f"On a {length}-game losing streak")

    # Home/Road splits
    def pct(stats):
        return stats["wins"] / stats["games"] if stats["games"] > 0 else 0.0

    home_home = home_profile["home"]
    away_road = away_profile["away"]
    home_home_win_pct = pct(home_home)
    away_road_win_pct = pct(away_road)
    if home_home["games"] > 0:
        home_score += home_home_win_pct * 1.2
        if home_home_win_pct >= 0.55:
            reasons_home.append("Strong home record")
        elif home_home_win_pct <= 0.45:
            reasons_away.append("Home team weaker at home")
    if away_road["games"] > 0:
        away_score += away_road_win_pct * 1.2
        if away_road_win_pct >= 0.55:
            reasons_away.append("Strong road record")
        elif away_road_win_pct <= 0.45:
            reasons_home.append("Away team weaker on the road")

    # Defensive edge (lower GA last 10)
    home_ga10 = home_profile["last10"]["avg_against"]
    away_ga10 = away_profile["last10"]["avg_against"]
    if home_ga10 + 0.5 < away_ga10:
        home_score += 1.0
        reasons_home.append("Better defensive numbers (fewer goals against)")
    elif away_ga10 + 0.5 < home_ga10:
        away_score += 1.0
        reasons_away.append("Better defensive numbers (fewer goals against)")

    return home_score, away_score, {
        "home_reasons": reasons_home,
        "away_reasons": reasons_away,
    }


def _ev_block(home_team, away_team, home_score, away_score, force_refresh=False):
    """Return EV calculations if odds available; else None."""
    odds = load_current_odds_for_matchup(
        home_team, away_team, force_refresh=force_refresh
    )
    if odds is None:
        logger.info("[EV] No odds for %s vs %s", home_team, away_team)
        return None

    home_odds = odds.get("home_ml")
    away_odds = odds.get("away_ml")
    if home_odds is None or away_odds is None:
        logger.info("[EV] Odds missing fields for %s vs %s", home_team, away_team)
        return None

    p_home_model, p_away_model = model_probs_from_scores(home_score, away_score)
    p_home_market = implied_prob_american(home_odds)
    p_away_market = implied_prob_american(away_odds)
    home_profit = profit_on_win_for_1_unit(home_odds)
    away_profit = profit_on_win_for_1_unit(away_odds)

    home_ev = p_home_model * home_profit - (1 - p_home_model)
    away_ev = p_away_model * away_profit - (1 - p_away_model)
    home_edge_pct = (p_home_model - p_home_market) * 100.0
    away_edge_pct = (p_away_model - p_away_market) * 100.0

    return {
        "odds": {"home_ml": home_odds, "away_ml": away_odds},
        "model_prob": {"home": p_home_model, "away": p_away_model},
        "market_prob": {"home": p_home_market, "away": p_away_market},
        "edge_pct": {"home": home_edge_pct, "away": away_edge_pct},
        "ev_units": {"home": home_ev, "away": away_ev},
        "grades": {
            "home": confidence_grade(home_edge_pct, home_ev),
            "away": confidence_grade(away_edge_pct, away_ev),
        },
    }


def _profile_summary(profile):
    """Select only summary fields for API output."""
    return {
        "team": profile["team"],
        "full": profile["full"],
        "last5": profile["last5"],
        "last10": profile["last10"],
        "home": profile["home"],
        "away": profile["away"],
        "high_scoring_last5": profile["high_scoring_last5"],
        "streak": profile["streak"],
    }


# ---------- Routes ----------


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info("REQUEST %s %s", request.method, request.url.path)
    response = await call_next(request)
    logger.info("RESPONSE %s %s -> %s", request.method, request.url.path, response.status_code)
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/sports")
def sports():
    """List supported sports and team codes (multisport groundwork)."""
    return {
        "sports": [
            {"id": key, "name": value["title"], "teams": value["teams"]}
            for key, value in SPORTS.items()
        ]
    }


@app.get("/teams")
def list_teams():
    """List NHL team abbreviations for UI dropdowns."""
    teams = SPORTS["nhl"]["teams"]
    return {"sport": "nhl", "teams": teams}


@app.get("/nhl/today")
def nhl_today(force_refresh: bool = Query(False)):
    """Return today's NHL schedule (home/away abbreviations)."""
    today = date.today().isoformat()
    schedule = load_schedule_for_date(today, force_refresh=force_refresh)
    logger.info(
        "[TODAY] Games: %s (force_refresh=%s)",
        len(schedule),
        force_refresh,
    )
    return {
        "date": today,
        "games": [
            {"home": g["home_team"], "away": g["away_team"]}
            for g in schedule
        ],
        "force_refresh": force_refresh,
    }


@app.get("/", include_in_schema=False)
def root():
    """Serve the quick test UI."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "UI not found. Ensure static/index.html exists."}


@app.get("/nhl/matchup")
def nhl_matchup(
    home: str = Query(..., description="Home team abbreviation, e.g. BOS"),
    away: str = Query(..., description="Away team abbreviation, e.g. TOR"),
    start_date: Optional[str] = Query(
        None, description="YYYY-MM-DD; defaults to 60 days ago"
    ),
    end_date: Optional[str] = Query(
        None, description="YYYY-MM-DD; defaults to today"
    ),
    force_refresh: bool = Query(
        False,
        description="If true, bypass cached ESPN responses for this request.",
    ),
):
    # Defaults: last 60 days to keep fetches fast
    today = date.today()
    default_start = today - timedelta(days=60)
    start_dt = _parse_date_or_400(start_date, default_start, "start_date")
    end_dt = _parse_date_or_400(end_date, today, "end_date")

    if end_dt < start_dt:
        raise HTTPException(
            status_code=400,
            detail="end_date must be on or after start_date.",
        )

    start_str = start_dt.isoformat()
    end_str = end_dt.isoformat()

    try:
        games = load_games_from_espn_date_range(
            start_str, end_str, force_refresh=force_refresh
        )
    except Exception as e:
        logger.exception("Data fetch failed for %s-%s | %s", start_str, end_str, e)
        raise HTTPException(status_code=502, detail=f"Data fetch failed: {e}")

    if not games:
        logger.info(
            "[MATCHUP] No games for %s vs %s in range %s-%s",
            home,
            away,
            start_str,
            end_str,
        )
        raise HTTPException(
            status_code=400,
            detail=(
                "No games available for the chosen date range. "
                "Try widening the dates."
            ),
        )

    home = _validate_team_or_400(home, "home")
    away = _validate_team_or_400(away, "away")

    home_profile = get_team_profile(home, games)
    away_profile = get_team_profile(away, games)
    if home_profile is None or away_profile is None:
        logger.info(
            "[MATCHUP] Insufficient data for %s or %s in range %s-%s",
            home,
            away,
            start_str,
            end_str,
        )
        raise HTTPException(
            status_code=400,
            detail="Not enough data for one or both teams in this range.",
        )

    try:
        home_score, away_score, reasons = _lean_scores(home, away, games)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    ev = _ev_block(home, away, home_score, away_score, force_refresh=force_refresh)

    diff = home_score - away_score
    side_lean = None
    if diff > 1.5:
        side_lean = f"{home} ML (strong lean)"
    elif diff > 0.5:
        side_lean = f"{home} ML (slight lean)"
    elif diff < -1.5:
        side_lean = f"{away} ML (strong lean)"
    elif diff < -0.5:
        side_lean = f"{away} ML (slight lean)"
    else:
        side_lean = "No clear edge"

    return {
        "params": {
            "home": home,
            "away": away,
            "start_date": start_str,
            "end_date": end_str,
            "force_refresh": force_refresh,
        },
        "side": {
            "home_score": home_score,
            "away_score": away_score,
            "lean": side_lean,
            "reasons": reasons,
        },
        "ev": ev,
        "profiles": {
            "home": _profile_summary(home_profile),
            "away": _profile_summary(away_profile),
        },
    }


@app.get("/nhl/team")
def nhl_team(
    team: str = Query(..., description="Team abbreviation, e.g. BOS"),
    start_date: Optional[str] = Query(
        None, description="YYYY-MM-DD; defaults to 60 days ago"
    ),
    end_date: Optional[str] = Query(
        None, description="YYYY-MM-DD; defaults to today"
    ),
    force_refresh: bool = Query(
        False,
        description="If true, bypass cached ESPN responses for this request.",
    ),
):
    today = date.today()
    default_start = today - timedelta(days=60)
    start_dt = _parse_date_or_400(start_date, default_start, "start_date")
    end_dt = _parse_date_or_400(end_date, today, "end_date")

    if end_dt < start_dt:
        raise HTTPException(
            status_code=400,
            detail="end_date must be on or after start_date.",
        )

    start_str = start_dt.isoformat()
    end_str = end_dt.isoformat()

    try:
        games = load_games_from_espn_date_range(
            start_str, end_str, force_refresh=force_refresh
        )
    except Exception as e:
        logger.exception("Data fetch failed for team %s | %s", team, e)
        raise HTTPException(status_code=502, detail=f"Data fetch failed: {e}")

    if not games:
        logger.info(
            "[TEAM] No games for %s in range %s-%s",
            team,
            start_str,
            end_str,
        )
        raise HTTPException(
            status_code=400,
            detail=(
                "No games available for the chosen date range. "
                "Try widening the dates."
            ),
        )

    team = _validate_team_or_400(team, "team")
    profile = get_team_profile(team, games)
    if profile is None:
        raise HTTPException(
            status_code=400,
            detail="Not enough data for that team in this range.",
        )

    return {
        "params": {
            "team": team,
            "start_date": start_str,
            "end_date": end_str,
            "force_refresh": force_refresh,
        },
        "profile": _profile_summary(profile),
    }
