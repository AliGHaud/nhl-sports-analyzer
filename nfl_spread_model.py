"""
Heuristic NFL spread model: predicts margin and cover probabilities using existing signals.
This is a lightweight starting point; intended to be refined/tuned in later phases.
"""

from __future__ import annotations

import math
from typing import Optional, Dict

from nfl_analyzer import (
    get_team_games,
    calculate_stats,
    get_rest_advantage,
    model_probs_from_scores,  # reuse temperature scaling if needed
)

try:
    from nfl_advanced_stats import get_team_epa, get_qb_status

    NFL_ADVANCED_STATS_AVAILABLE = True
except Exception:
    NFL_ADVANCED_STATS_AVAILABLE = False


def american_to_decimal(odds: float) -> float:
    odds = float(odds)
    if odds < 0:
        return 1 + (100.0 / abs(odds))
    return 1 + (odds / 100.0)


def predict_margin(
    home_team: str,
    away_team: str,
    games: list[dict],
    game_date: Optional[str] = None,
    season: Optional[int] = None,
) -> Optional[float]:
    """
    Predict expected margin (home score - away score) using simple signals.
    """
    home_profile = calculate_stats(home_team, get_team_games(games, home_team))
    away_profile = calculate_stats(away_team, get_team_games(games, away_team))
    if home_profile["games"] == 0 or away_profile["games"] == 0:
        return None

    margin = 2.5  # base home field in points

    # Point differential signal
    home_pt_diff = home_profile["pf"] - home_profile["pa"]
    away_pt_diff = away_profile["pf"] - away_profile["pa"]
    avg_home_margin = home_pt_diff / home_profile["games"] if home_profile["games"] > 0 else 0
    avg_away_margin = away_pt_diff / away_profile["games"] if away_profile["games"] > 0 else 0
    margin += (avg_home_margin - avg_away_margin) * 0.5  # dampen to avoid extremes

    # Rest advantage
    if game_date:
        rest_diff, home_rest, away_rest = get_rest_advantage(home_team, away_team, games, game_date)
        if rest_diff >= 3:
            margin += 1.0
        elif rest_diff <= -3:
            margin -= 1.0
        if home_rest is not None and home_rest <= 4:
            margin -= 0.5
        if away_rest is not None and away_rest <= 4:
            margin += 0.5

    # Advanced stats (EPA) if available
    if season and NFL_ADVANCED_STATS_AVAILABLE:
        try:
            home_epa = get_team_epa(home_team, season)
            away_epa = get_team_epa(away_team, season)
            if home_epa and away_epa:
                epa_diff = home_epa["net_epa"] - away_epa["net_epa"]
                margin += epa_diff * 6.0  # scale EPA/play to points
            home_qb = get_qb_status(home_team, season)
            away_qb = get_qb_status(away_team, season)
            if home_qb and home_qb.get("is_backup"):
                margin -= 3.0
            if away_qb and away_qb.get("is_backup"):
                margin += 3.0
        except Exception:
            pass

    return margin


def _std_normal_cdf(x: float) -> float:
    """Standard normal CDF via erf."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def cover_probabilities(margin_pred: float, home_spread: float, std: float = 13.5) -> Dict[str, float]:
    """
    Return cover probabilities for home/away given predicted margin and home spread.
    Positive home spread means home is underdog.
    """
    z = (home_spread - margin_pred) / std
    home_cover = 1 - _std_normal_cdf(z)
    away_cover = 1 - home_cover
    return {"home": home_cover, "away": away_cover}


def spread_ev(
    margin_pred: float,
    home_spread: float,
    home_odds: float = -110,
    away_odds: float = -110,
    std: float = 13.5,
) -> Dict[str, Dict[str, float]]:
    """
    Compute cover probabilities and EV for each side given spreads/odds.
    """
    probs = cover_probabilities(margin_pred, home_spread, std=std)
    home_dec = american_to_decimal(home_odds)
    away_dec = american_to_decimal(away_odds)
    home_ev = probs["home"] * (home_dec - 1.0) - (1 - probs["home"])
    away_ev = probs["away"] * (away_dec - 1.0) - (1 - probs["away"])
    return {
        "prob": probs,
        "ev_units": {"home": home_ev, "away": away_ev},
    }
