"""Lightweight FastAPI wrapper around the NHL analyzer logic.

Endpoints:
- GET /health
- GET /nhl/matchup?home=TOR&away=BOS[&start_date=YYYY-MM-DD&end_date=YYYY-MM-DD]
  Fetches recent games (default last 45 days), builds team profiles, runs lean + EV,
  and returns JSON suitable for a front-end.
"""

import logging
import os
import json
from datetime import date, datetime, timedelta, time, timezone
from pathlib import Path
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from data_sources import (
    CACHE_DIR,
    load_games_from_espn_date_range,
    load_current_odds_for_matchup,
    load_schedule_for_date,
    TEAM_CODE_TO_ID,
    read_pick_cache,
    write_pick_cache,
    get_probable_goalie,
    get_team_adv_stats,
    load_nhl_team_stats,
    load_moneypuck_team_stats,
    load_team_roster,
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

APP_TIMEZONE = os.getenv("APP_TIMEZONE", "America/New_York")
PICK_GATE_HOUR = int(os.getenv("PICK_GATE_HOUR", "12"))
PICK_GATE_MINUTE = int(os.getenv("PICK_GATE_MINUTE", "0"))
DEFAULT_LOOKBACK_DAYS = int(os.getenv("DEFAULT_LOOKBACK_DAYS", "45"))
API_ADMIN_TOKEN = os.getenv("API_ADMIN_TOKEN")
MANUAL_OVERRIDES_PATH = CACHE_DIR / "manual_overrides.json"


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


def _now_in_zone():
    """Timezone-aware now in the configured app timezone."""
    try:
        zone = ZoneInfo(APP_TIMEZONE)
    except Exception:
        try:
            zone = ZoneInfo("UTC")
        except Exception:
            # Fall back to naive UTC if tzdata is missing
            return datetime.now(timezone.utc)
    return datetime.now(zone)


def _read_overrides():
    if not MANUAL_OVERRIDES_PATH.exists():
        return {"injuries": [], "goalies": []}
    try:
        with MANUAL_OVERRIDES_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"injuries": [], "goalies": []}
        data.setdefault("injuries", [])
        data.setdefault("goalies", [])
        return data
    except Exception:
        return {"injuries": [], "goalies": []}


def _write_overrides(data: dict):
    try:
        MANUAL_OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with MANUAL_OVERRIDES_PATH.open("w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def _require_admin(request: Request):
    if not API_ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Admin token not configured.")
    token = request.headers.get("x-admin-token") or request.headers.get("X-Admin-Token")
    if token != API_ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _lean_scores(home_team: str, away_team: str, games, force_refresh: bool = False) -> Tuple[float, float, dict]:
    """Pure version of the lean logic: returns scores + reasons."""
    today = _now_in_zone().date()
    home_profile = get_team_profile(home_team, games, today=today)
    away_profile = get_team_profile(away_team, games, today=today)
    adv_home = get_team_adv_stats(home_team, force_refresh=force_refresh) or {}
    adv_away = get_team_adv_stats(away_team, force_refresh=force_refresh) or {}

    if home_profile is None or away_profile is None:
        raise ValueError("Not enough data to build team profiles.")

    def win_pct(stats):
        g = stats["games"]
        return stats["wins"] / g if g > 0 else 0.0

    home_score = 0.0
    away_score = 0.0
    reasons_home = []
    reasons_away = []

    # Recency: use last 10 record as a single modest factor
    home_win10 = win_pct(home_profile["last10"])
    away_win10 = win_pct(away_profile["last10"])
    home_score += home_win10 * 1.0
    away_score += away_win10 * 1.0
    if home_win10 > away_win10 + 0.1:
        reasons_home.append("Better recent form (last 10)")
    elif away_win10 > home_win10 + 0.1:
        reasons_away.append("Better recent form (last 10)")

    # Streaks: light and capped
    for profile, is_home_flag in ((home_profile, True), (away_profile, False)):
        streak = profile["streak"]
        length = streak["length"]
        stype = streak["type"]
        if length > 0 and stype:
            delta_raw = 0.25 * length if stype == "W" else -0.25 * length
            delta = max(min(delta_raw, 0.75), -0.75)
            if is_home_flag:
                home_score += delta
                if stype == "W" and length >= 3:
                    reasons_home.append(f"On a {length}-game winning streak")
                if stype == "L" and length >= 3:
                    reasons_home.append(f"On a {length}-game losing streak")
            else:
                away_score += delta
                if stype == "W" and length >= 3:
                    reasons_away.append(f"On a {length}-game winning streak")
                if stype == "L" and length >= 3:
                    reasons_away.append(f"On a {length}-game losing streak")

    # Flat home-ice bonus
    home_score += 0.75
    reasons_home.append("Home-ice advantage")

    # Home/Road splits
    def pct(stats):
        return stats["wins"] / stats["games"] if stats["games"] > 0 else 0.0

    home_home = home_profile["home"]
    away_road = away_profile["away"]
    home_home_win_pct = pct(home_home)
    away_road_win_pct = pct(away_road)
    if home_home["games"] > 0:
        home_score += home_home_win_pct * 0.8
        if home_home_win_pct >= 0.55:
            reasons_home.append("Strong home record")
        elif home_home_win_pct <= 0.45:
            reasons_away.append("Home team weaker at home")
    if away_road["games"] > 0:
        away_score += away_road_win_pct * 0.8
        if away_road_win_pct >= 0.55:
            reasons_away.append("Strong road record")
        elif away_road_win_pct <= 0.45:
            reasons_home.append("Away team weaker on the road")

    # Defensive edge (lower GA last 10) with modest weight
    home_ga10 = home_profile["last10"]["avg_against"]
    away_ga10 = away_profile["last10"]["avg_against"]
    if home_ga10 + 0.5 < away_ga10:
        home_score += 0.5
        reasons_home.append("Better defensive numbers (fewer goals against)")
    elif away_ga10 + 0.5 < home_ga10:
        away_score += 0.5
        reasons_away.append("Better defensive numbers (fewer goals against)")

    # Advanced stats edge (xG/HDCF) from MoneyPuck if available
    xgf_pct_home = adv_home.get("xgf_pct")
    xgf_pct_away = adv_away.get("xgf_pct")
    if xgf_pct_home is not None and xgf_pct_away is not None:
        delta = xgf_pct_home - xgf_pct_away
        if delta >= 3.0:
            home_score += 1.0
            reasons_home.append("Better xG share (season-to-date)")
        elif delta <= -3.0:
            away_score += 1.0
            reasons_away.append("Better xG share (season-to-date)")

    # Special teams edge (PP/PK) if meaningful
    pp_home = adv_home.get("pp")
    pk_home = adv_home.get("pk")
    pp_away = adv_away.get("pp")
    pk_away = adv_away.get("pk")
    if pp_home and pk_home and pp_away and pk_away:
        # simple net special teams strength
        net_home = (pp_home or 0) - (100 - (pk_home or 0))
        net_away = (pp_away or 0) - (100 - (pk_away or 0))
        delta = net_home - net_away
        if delta >= 5:
            home_score += 0.3
            reasons_home.append("Special teams edge")
        elif delta <= -5:
            away_score += 0.3
            reasons_away.append("Special teams edge")

    # Rest / back-to-back
    rest_home = home_profile.get("rest") or {}
    rest_away = away_profile.get("rest") or {}
    home_days = rest_home.get("days_since_last")
    away_days = rest_away.get("days_since_last")
    if rest_home.get("is_back_to_back") and not rest_away.get("is_back_to_back"):
        home_score -= 0.5
        reasons_away.append("Home on back-to-back; away more rested")
    if rest_away.get("is_back_to_back") and not rest_home.get("is_back_to_back"):
        away_score -= 0.5
        reasons_home.append("Away on back-to-back; home more rested")
    if home_days is not None and away_days is not None:
        diff = home_days - away_days
        if diff >= 2:
            home_score += 0.4
            reasons_home.append(f"More rest ({home_days}d vs {away_days}d)")
        elif diff <= -2:
            away_score += 0.4
            reasons_away.append(f"More rest ({away_days}d vs {home_days}d)")

    # Goalie edge (heuristic)
    try:
        home_g = get_probable_goalie(home_team, force_refresh=force_refresh)
        away_g = get_probable_goalie(away_team, force_refresh=force_refresh)
    except Exception:
        home_g = None
        away_g = None

    if home_g and away_g:
        home_sv = home_g["stats"].get("save_pct")
        away_sv = away_g["stats"].get("save_pct")
        if home_sv is not None and away_sv is not None:
            delta = home_sv - away_sv
            if delta >= 0.01:
                home_score += 1.2
                reasons_home.append(f"Goalie edge: {home_g['name']} higher sv%")
            elif delta <= -0.01:
                away_score += 1.2
                reasons_away.append(f"Goalie edge: {away_g['name']} higher sv%")

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
        "rest": profile.get("rest"),
    }


def _team_stats_snapshot(force_refresh: bool = False):
    nhl_stats = load_nhl_team_stats(force_refresh=force_refresh)
    adv_stats = load_moneypuck_team_stats(force_refresh=force_refresh)
    teams = []
    all_codes = set(nhl_stats.keys()) | set(adv_stats.keys())
    for code in sorted(all_codes):
        teams.append(
            {
                "team": code,
                "nhl": nhl_stats.get(code) or {},
                "adv": adv_stats.get(code) or {},
            }
        )
    return teams


class InjuryEntry(BaseModel):
    team: str
    player: str
    status: Optional[str] = None
    note: Optional[str] = None
    probable_goalie: Optional[bool] = None
    updated_at: Optional[str] = None


class GoalieEntry(BaseModel):
    team: str
    goalie: str
    expected_starter: bool = False
    updated_at: Optional[str] = None


class OverridePayload(BaseModel):
    injuries: list[InjuryEntry] = []
    goalies: list[GoalieEntry] = []

# ---------- Pick of the Day helpers ----------

def _pick_candidate_from_ev(home, away, ev, reasons):
    """Return the best side candidate from an EV block or None."""
    if ev is None:
        return None

    best = None
    for side_key, team in (("home", home), ("away", away)):
        ev_units = ev["ev_units"][side_key]
        edge_pct = ev["edge_pct"][side_key]
        grade = ev["grades"][side_key]
        model_prob = ev["model_prob"][side_key]
        market_prob = ev["market_prob"][side_key]
        odds = ev["odds"][f"{side_key}_ml"]

        # Thresholds for a viable pick
        if ev_units < 0.03:
            continue
        if edge_pct < 2.0:
            continue
        if model_prob < 0.52:
            continue
        if grade == "No Bet":
            continue

        candidate = {
            "matchup": {"home": home, "away": away},
            "side": team,
            "side_type": side_key,
            "odds": odds,
            "ev_units": ev_units,
            "edge_pct": edge_pct,
            "grade": grade,
            "model_prob": model_prob,
            "market_prob": market_prob,
            "reasons": reasons["home_reasons"] if side_key == "home" else reasons["away_reasons"],
        }
        if best is None or candidate["ev_units"] > best["ev_units"]:
            best = candidate
    return best


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
    now = _now_in_zone()
    today = now.date().isoformat()
    schedule = load_schedule_for_date(today, force_refresh=force_refresh)
    enriched = []
    for g in schedule:
        odds = load_current_odds_for_matchup(
            g["home_team"], g["away_team"], force_refresh=force_refresh
        )
        enriched.append(
            {
                "home_team": g["home_team"],
                "away_team": g["away_team"],
                "odds": odds,
            }
        )
    logger.info(
        "[TODAY] Games: %s (force_refresh=%s)",
        len(enriched),
        force_refresh,
    )
    return {
        "date": today,
        "timezone": APP_TIMEZONE,
        "games": [
            {"home": g["home_team"], "away": g["away_team"], "odds": g["odds"]}
            for g in enriched
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


@app.get("/admin", include_in_schema=False)
def admin_console():
    """Serve the admin overrides UI."""
    admin_path = STATIC_DIR / "admin.html"
    if admin_path.exists():
        return FileResponse(admin_path)
    return {"message": "Admin UI not found. Ensure static/admin.html exists."}


@app.get("/nhl/matchup")
def nhl_matchup(
    home: str = Query(..., description="Home team abbreviation, e.g. BOS"),
    away: str = Query(..., description="Away team abbreviation, e.g. TOR"),
    start_date: Optional[str] = Query(
        None, description="YYYY-MM-DD; defaults to 45 days ago"
    ),
    end_date: Optional[str] = Query(
        None, description="YYYY-MM-DD; defaults to today"
    ),
    force_refresh: bool = Query(
        False,
        description="If true, bypass cached ESPN responses for this request.",
    ),
    ):
    # Defaults: last 45 days to keep fetches fast and recent
    now = _now_in_zone()
    today = now.date()
    default_start = today - timedelta(days=DEFAULT_LOOKBACK_DAYS)
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

    today = _now_in_zone().date()
    home_profile = get_team_profile(home, games, today=today)
    away_profile = get_team_profile(away, games, today=today)
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
        home_score, away_score, reasons = _lean_scores(home, away, games, force_refresh=force_refresh)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Always expose model win probabilities, even if odds/EV are unavailable.
    model_home_prob, model_away_prob = model_probs_from_scores(home_score, away_score)
    model_probs = {"home": model_home_prob, "away": model_away_prob}

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
        "model_prob": model_probs,
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
        None, description="YYYY-MM-DD; defaults to 45 days ago"
    ),
    end_date: Optional[str] = Query(
        None, description="YYYY-MM-DD; defaults to today"
    ),
    force_refresh: bool = Query(
        False,
        description="If true, bypass cached ESPN responses for this request.",
    ),
):
    now = _now_in_zone()
    today = now.date()
    default_start = today - timedelta(days=DEFAULT_LOOKBACK_DAYS)
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
    today = _now_in_zone().date()
    profile = get_team_profile(team, games, today=today)
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


@app.get("/nhl/pick")
def nhl_pick(
    lookback_days: int = Query(
        DEFAULT_LOOKBACK_DAYS,
        ge=7,
        le=200,
        description="Days of history to use for profiles (min 7, max 200).",
    ),
    force_refresh: bool = Query(
        False,
        description="If true, bypass cached ESPN responses for this request.",
    ),
    ignore_pick_gate: bool = Query(
        False,
        description="If true, skip the noon gating (testing/internal).",
    ),
    cache: bool = Query(
        True,
        description="If true, returns cached pick for the day when available.",
    ),
):
    now = _now_in_zone()
    today = now.date()
    start_dt = today - timedelta(days=lookback_days)
    start_str = start_dt.isoformat()
    end_str = today.isoformat()

    pick_gate = time(PICK_GATE_HOUR, PICK_GATE_MINUTE)
    if cache and not force_refresh:
        cached = read_pick_cache(end_str)
        if cached:
            cached["cached"] = True
            return cached

    if not ignore_pick_gate and now.time() < pick_gate:
        return {
            "date": end_str,
            "timezone": APP_TIMEZONE,
            "pick": None,
            "reason": f"Pick available after {pick_gate.strftime('%I:%M %p')} {APP_TIMEZONE}.",
            "candidates": [],
            "params": {
                "lookback_days": lookback_days,
                "force_refresh": force_refresh,
                "ignore_pick_gate": ignore_pick_gate,
            },
        }

    schedule = load_schedule_for_date(end_str, force_refresh=force_refresh)
    if not schedule:
        return {
            "date": end_str,
            "timezone": APP_TIMEZONE,
            "pick": None,
            "reason": "No games scheduled today.",
            "candidates": [],
            "params": {"lookback_days": lookback_days, "force_refresh": force_refresh},
        }

    try:
        games = load_games_from_espn_date_range(
            start_str, end_str, force_refresh=force_refresh
        )
    except Exception as e:
        logger.exception("Data fetch failed for pick of day %s-%s | %s", start_str, end_str, e)
        raise HTTPException(status_code=502, detail=f"Data fetch failed: {e}")

    if not games:
        return {
            "date": end_str,
            "timezone": APP_TIMEZONE,
            "pick": None,
            "reason": "No historical games available to build profiles.",
            "candidates": [],
            "params": {"lookback_days": lookback_days, "force_refresh": force_refresh},
        }

    candidates = []
    for g in schedule:
        home = g["home_team"]
        away = g["away_team"]
        try:
            home_score, away_score, reasons = _lean_scores(home, away, games, force_refresh=force_refresh)
        except ValueError:
            continue

        ev = _ev_block(home, away, home_score, away_score, force_refresh=force_refresh)
        candidate = _pick_candidate_from_ev(home, away, ev, reasons)
        if candidate:
            candidates.append(candidate)

    if not candidates:
        return {
            "date": end_str,
            "timezone": APP_TIMEZONE,
            "pick": None,
            "reason": "No qualifying pick met the thresholds.",
            "candidates": [],
            "params": {"lookback_days": lookback_days, "force_refresh": force_refresh},
        }

    best = max(candidates, key=lambda c: c["ev_units"])
    result = {
        "date": end_str,
    "timezone": APP_TIMEZONE,
    "pick": best,
    "candidates": candidates,
    "params": {"lookback_days": lookback_days, "force_refresh": force_refresh},
    }
    if cache:
        try:
            write_pick_cache(end_str, result)
        except Exception:
            pass
    return result


@app.get("/nhl/stats")
def nhl_stats(force_refresh: bool = Query(False, description="Refresh cached stats sources")):
    """
    Return merged team stats from NHL API (PP/PK/FO/shots) and MoneyPuck (xG/HDCF/PP/PK).
    """
    teams = _team_stats_snapshot(force_refresh=force_refresh)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "force_refresh": force_refresh,
        "teams": teams,
    }


@app.get("/nhl/roster")
def nhl_roster(
    team: str = Query(..., description="Team abbreviation, e.g. BOS"),
    force_refresh: bool = Query(False, description="Refresh roster cache"),
):
    team = _validate_team_or_400(team, "team")
    roster = load_team_roster(team, force_refresh=force_refresh)
    if not roster:
        raise HTTPException(status_code=404, detail="Roster not available for that team.")
    players = []
    try:
        athletes = roster.get("athletes", [])
    except Exception:
        athletes = []
    for group in athletes:
        try:
            position = group.get("position", {}).get("abbreviation")
        except Exception:
            position = None
        for item in group.get("items", []):
            try:
                name = item.get("fullName") or item.get("displayName")
            except Exception:
                name = None
            if not name:
                continue
            players.append({"name": name, "position": position})
    return {"team": team, "players": players}


@app.get("/admin/overrides")
def get_overrides(request: Request):
    _require_admin(request)
    return _read_overrides()


@app.post("/admin/overrides")
def update_overrides(payload: OverridePayload, request: Request):
    _require_admin(request)
    data = {
        "injuries": [entry.dict() for entry in payload.injuries],
        "goalies": [entry.dict() for entry in payload.goalies],
    }
    _write_overrides(data)
    return {"status": "ok", "overrides": data}
