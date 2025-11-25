"""
Simple NFL lean engine (separate from NHL).
"""

from datetime import date
import math

from nfl_data_sources import (
    load_games_from_espn_scoreboard,
    load_games_from_espn_date_range,
    load_current_odds_for_matchup,
    load_schedule_for_date,
)


def get_team_games(games, team):
    return [g for g in games if g["home_team"] == team or g["away_team"] == team]


def get_last_n_games(team, games, n=5):
    tg = get_team_games(games, team)
    return tg[-n:] if len(tg) >= n else tg


def calculate_stats(team, games):
    wins = 0
    losses = 0
    pf = pa = 0
    for g in games:
        hg = int(g["home_goals"])
        ag = int(g["away_goals"])
        if g["home_team"] == team:
            pf += hg
            pa += ag
            wins += hg > ag
            losses += hg < ag
        else:
            pf += ag
            pa += hg
            wins += ag > hg
            losses += ag < hg
    n = len(games)
    return {
        "games": n,
        "wins": wins,
        "losses": losses,
        "pf": pf,
        "pa": pa,
        "avg_pf": pf / n if n else 0,
        "avg_pa": pa / n if n else 0,
    }


def implied_prob_american(odds):
    odds = float(odds)
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)


def profit_on_win_for_1_unit(odds):
    odds = float(odds)
    if odds < 0:
        return 100.0 / abs(odds)
    return odds / 100.0


def model_probs_from_scores(home_score, away_score, temperature=1.15):
    t = max(float(temperature), 1e-6)
    eh = math.exp(home_score / t)
    ea = math.exp(away_score / t)
    total = eh + ea
    return eh / total, ea / total


def confidence_grade(edge_pct, ev_units):
    if ev_units <= 0:
        return "No Bet"
    edge = abs(edge_pct)
    if edge >= 7.5 and ev_units >= 0.10:
        return "A"
    if edge >= 5.0 and ev_units >= 0.06:
        return "B+"
    if edge >= 3.0 and ev_units >= 0.03:
        return "B"
    if edge >= 1.5 and ev_units >= 0.01:
        return "C+"
    if edge >= 0.5 and ev_units > 0:
        return "C"
    return "D"


def lean_matchup(home_team, away_team, games):
    home_profile = calculate_stats(home_team, get_team_games(games, home_team))
    away_profile = calculate_stats(away_team, get_team_games(games, away_team))
    if home_profile["games"] == 0 or away_profile["games"] == 0:
        return None, None, {"home_reasons": [], "away_reasons": []}

    home_score = 0.0
    away_score = 0.0
    reasons_home = []
    reasons_away = []

    def win_pct(stats):
        return stats["wins"] / stats["games"] if stats["games"] > 0 else 0.0

    # Last 5 win pct
    home_last5 = calculate_stats(home_team, get_last_n_games(home_team, games, 5))
    away_last5 = calculate_stats(away_team, get_last_n_games(away_team, games, 5))
    home_score += win_pct(home_last5) * 1.0
    away_score += win_pct(away_last5) * 1.0
    if win_pct(home_last5) > win_pct(away_last5) + 0.1:
        reasons_home.append("Better recent form (last 5)")
    elif win_pct(away_last5) > win_pct(home_last5) + 0.1:
        reasons_away.append("Better recent form (last 5)")

    # Points allowed (defense)
    if home_last5["avg_pa"] + 2 < away_last5["avg_pa"]:
        home_score += 0.5
        reasons_home.append("Better defense (points allowed)")
    elif away_last5["avg_pa"] + 2 < home_last5["avg_pa"]:
        away_score += 0.5
        reasons_away.append("Better defense (points allowed)")

    # Points for (offense)
    if home_last5["avg_pf"] > away_last5["avg_pf"] + 2:
        home_score += 0.5
        reasons_home.append("Better offense (scoring)")
    elif away_last5["avg_pf"] > home_last5["avg_pf"] + 2:
        away_score += 0.5
        reasons_away.append("Better offense (scoring)")

    # Home field (light)
    home_score += 0.3
    reasons_home.append("Home field")

    diff = home_score - away_score
    lean = "No clear edge"
    if diff > 1.0:
        lean = f"{home_team} ML (strong lean)"
    elif diff > 0.3:
        lean = f"{home_team} ML (slight lean)"
    elif diff < -1.0:
        lean = f"{away_team} ML (strong lean)"
    elif diff < -0.3:
        lean = f"{away_team} ML (slight lean)"

    return home_score, away_score, {"home_reasons": reasons_home, "away_reasons": reasons_away, "lean": lean}
