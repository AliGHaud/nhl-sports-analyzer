# nhl_analyzer.py
# Main file for your NHL betting analyzer project.

from pathlib import Path
from datetime import date, timedelta, datetime
import math

from data_sources import (
    load_games_from_espn_scoreboard,
    load_games_from_espn_date_range,
    load_current_odds_for_matchup,
    load_schedule_for_date,  # future schedule helper
    get_team_adv_stats,
)

# Points to the folder where this script lives (not heavily used now, but kept)
BASE_DIR = Path(__file__).parent
DEFAULT_NHL_TEMPERATURE = 1.5
MIN_MODEL_PROBABILITY = 0.5
MIN_EDGE_FAVORITE = 0.05
MIN_EDGE_UNDERDOG = 0.22


# =========================
#   BASIC STATS HELPERS
# =========================

def get_team_games(games, team):
    """
    Return a list of all games where the team played.
    """
    team_games = []

    for g in games:
        if g["home_team"] == team or g["away_team"] == team:
            team_games.append(g)

    return team_games


def get_last_n_games(team_games, n=5):
    """
    Return the last n games for the team.
    Assumes the data is in chronological order.
    """
    return team_games[-n:] if len(team_games) >= n else team_games


def calculate_team_stats(team, games):
    """
    Calculate wins, losses, goals for/against, averages, etc.
    """
    wins = 0
    losses = 0
    goals_for = 0
    goals_against = 0

    for g in games:
        home_goals = int(g["home_goals"])
        away_goals = int(g["away_goals"])

        if g["home_team"] == team:
            gf = home_goals
            ga = away_goals
            won = home_goals > away_goals
        else:
            gf = away_goals
            ga = home_goals
            won = away_goals > home_goals

        goals_for += gf
        goals_against += ga

        if won:
            wins += 1
        else:
            losses += 1

    num_games = len(games)
    avg_for = goals_for / num_games if num_games > 0 else 0
    avg_against = goals_against / num_games if num_games > 0 else 0

    return {
        "team": team,
        "games": num_games,
        "wins": wins,
        "losses": losses,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "avg_for": avg_for,
        "avg_against": avg_against,
    }


def calculate_home_road_stats(team, team_games):
    """
    Calculate separate stats for:
    - Home games
    - Road (away) games

    Returns a dictionary like:
    {
        "home": { ... },
        "away": { ... }
    }
    """

    # Home stats
    home_games = 0
    home_wins = 0
    home_losses = 0
    home_goals_for = 0
    home_goals_against = 0

    # Away stats
    away_games = 0
    away_wins = 0
    away_losses = 0
    away_goals_for = 0
    away_goals_against = 0

    for g in team_games:
        home_goals = int(g["home_goals"])
        away_goals = int(g["away_goals"])

        if g["home_team"] == team:
            # HOME game
            home_games += 1
            home_goals_for += home_goals
            home_goals_against += away_goals

            if home_goals > away_goals:
                home_wins += 1
            else:
                home_losses += 1

        elif g["away_team"] == team:
            # AWAY (road) game
            away_games += 1
            away_goals_for += away_goals
            away_goals_against += home_goals

            if away_goals > home_goals:
                away_wins += 1
            else:
                away_losses += 1

    # Avoid division by zero
    home_avg_for = home_goals_for / home_games if home_games > 0 else 0
    home_avg_against = home_goals_against / home_games if home_games > 0 else 0

    away_avg_for = away_goals_for / away_games if away_games > 0 else 0
    away_avg_against = away_goals_against / away_games if away_games > 0 else 0

    return {
        "home": {
            "games": home_games,
            "wins": home_wins,
            "losses": home_losses,
            "goals_for": home_goals_for,
            "goals_against": home_goals_against,
            "avg_for": home_avg_for,
            "avg_against": home_avg_against,
        },
        "away": {
            "games": away_games,
            "wins": away_wins,
            "losses": away_losses,
            "goals_for": away_goals_for,
            "goals_against": away_goals_against,
            "avg_for": away_avg_for,
            "avg_against": away_avg_against,
        },
    }


def calculate_last_n_stats(team, team_games, n=5):
    """
    Calculate stats for the team's last n games.
    Uses the same logic as calculate_team_stats, but only on recent games.
    """
    last_games = get_last_n_games(team_games, n)
    return calculate_team_stats(team, last_games)


def count_high_scoring_games(team, team_games, n=5, threshold=3):
    """
    Count how many of the last n games this team scored at least `threshold` goals.
    """
    last_games = get_last_n_games(team_games, n)
    count = 0

    for g in last_games:
        home_goals = int(g["home_goals"])
        away_goals = int(g["away_goals"])

        if g["home_team"] == team:
            gf = home_goals
        else:
            gf = away_goals

        if gf >= threshold:
            count += 1

    return count, len(last_games)


def get_current_streak(team, team_games):
    """
    Look at the team's most recent games and figure out their current streak.

    Returns:
      (streak_length, streak_type)
      streak_type: "W" for winning, "L" for losing, or None
    """
    if not team_games:
        return 0, None

    streak_type = None
    streak_length = 0

    # Most recent to older
    for g in reversed(team_games):
        home_goals = int(g["home_goals"])
        away_goals = int(g["away_goals"])

        if g["home_team"] == team:
            won = home_goals > away_goals
        else:
            won = away_goals > home_goals

        this_type = "W" if won else "L"

        if streak_type is None:
            streak_type = this_type
            streak_length = 1
        elif this_type == streak_type:
            streak_length += 1
        else:
            break

    return streak_length, streak_type


def _compute_rest_info(team_games, today=None):
    """
    Compute days since last game and a simple back-to-back flag.
    """
    if not team_games:
        return {"days_since_last": None, "is_back_to_back": False}

    today = today or date.today()
    try:
        last_date = max(datetime.strptime(g["date"], "%Y-%m-%d").date() for g in team_games)
    except Exception:
        return {"days_since_last": None, "is_back_to_back": False}

    days = (today - last_date).days
    return {
        "days_since_last": days,
        "is_back_to_back": days <= 1,
    }


def get_team_profile(team, games, today=None):
    """
    Build a 'profile' for a team that gathers all the key stats we care about
    in one dictionary.

    today: optional date for rest calculations (defaults to date.today()).
    """
    team_games = get_team_games(games, team)
    if not team_games:
        return None

    full = calculate_team_stats(team, team_games)
    last5 = calculate_last_n_stats(team, team_games, n=5)
    last10 = calculate_last_n_stats(team, team_games, n=10)
    home_road = calculate_home_road_stats(team, team_games)
    hs_count_5, hs_total_5 = count_high_scoring_games(
        team, team_games, n=5, threshold=3
    )
    streak_len, streak_type = get_current_streak(team, team_games)
    rest_info = _compute_rest_info(team_games, today=today)

    return {
        "team": team,
        "games_list": team_games,
        "full": full,
        "last5": last5,
        "last10": last10,
        "home": home_road["home"],
        "away": home_road["away"],
        "high_scoring_last5": {
            "count": hs_count_5,
            "total": hs_total_5,
        },
        "streak": {
            "length": streak_len,
            "type": streak_type,
        },
        "rest": rest_info,
    }


# =========================
#   DISPLAY TEAM ANALYSIS
# =========================

def analyze_team(team, games):
    """
    Print a summary for a single team.
    """
    print(f"\n=== Analysis for {team} ===")

    team_games = get_team_games(games, team)

    if not team_games:
        print("No games found for this team in the data.")
        return

    # Full sample
    stats = calculate_team_stats(team, team_games)
    print("\nFull sample:")
    print(f"Games: {stats['games']}, Wins: {stats['wins']}, Losses: {stats['losses']}")
    print(
        f"Goals For: {stats['goals_for']}, "
        f"Goals Against: {stats['goals_against']}"
    )
    print(
        f"Avg Goals For: {stats['avg_for']:.2f}, "
        f"Avg Against: {stats['avg_against']:.2f}"
    )

    # Home vs Road splits
    home_road = calculate_home_road_stats(team, team_games)
    home = home_road["home"]
    away = home_road["away"]

    print("\nHome vs Road splits:")

    if home["games"] > 0:
        print(
            f"Home: {home['games']} games | "
            f"Record {home['wins']}-{home['losses']} | "
            f"Avg GF {home['avg_for']:.2f}, Avg GA {home['avg_against']:.2f}"
        )
    else:
        print("Home: no home games in this dataset.")

    if away["games"] > 0:
        print(
            f"Road: {away['games']} games | "
            f"Record {away['wins']}-{away['losses']} | "
            f"Avg GF {away['avg_for']:.2f}, Avg GA {away['avg_against']:.2f}"
        )
    else:
        print("Road: no away games in this dataset.")

    # Last 5 games
    print("\nLast 5 games:")
    last5 = calculate_last_n_stats(team, team_games, n=5)
    print(f"Games analyzed: {last5['games']}")
    print(f"Wins: {last5['wins']}, Losses: {last5['losses']}")
    print(
        f"Goals For: {last5['goals_for']}, "
        f"Goals Against: {last5['goals_against']}"
    )
    print(
        f"Avg Goals For: {last5['avg_for']:.2f}, "
        f"Avg Against: {last5['avg_against']:.2f}"
    )

    # Last 10 games
    print("\nLast 10 games:")
    last10 = calculate_last_n_stats(team, team_games, n=10)
    print(f"Games analyzed: {last10['games']}")
    print(f"Wins: {last10['wins']}, Losses: {last10['losses']}")
    print(
        f"Goals For: {last10['goals_for']}, "
        f"Goals Against: {last10['goals_against']}"
    )
    print(
        f"Avg Goals For: {last10['avg_for']:.2f}, "
        f"Avg Against: {last10['avg_against']:.2f}"
    )

    # High-scoring trend (3+ goals in last 5)
    threshold = 3
    high_scoring_count, num_last = count_high_scoring_games(
        team, team_games, n=5, threshold=threshold
    )
    print(
        f"\n{team} has scored {threshold}+ goals in "
        f"{high_scoring_count} of their last {num_last} games."
    )

    # Current streak
    streak_length, streak_type = get_current_streak(team, team_games)

    if streak_length == 0 or streak_type is None:
        print("Current streak: none (no games in data).")
    else:
        word = "winning" if streak_type == "W" else "losing"
        print(f"Current streak: {streak_length}-game {word} streak.")


# =========================
#   LEAN ENGINE + EV
# =========================

def implied_prob_american(odds):
    """
    Convert American odds to implied probability.

    Example:
      -135 -> about 57.4%
      +120 -> about 45.5%
    """
    odds = float(odds)
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)


def profit_on_win_for_1_unit(odds):
    """
    For a 1 unit bet, how much PROFIT do we make if the bet wins?

    Example:
      odds = -135  -> profit ~ 0.74 units on a 1u stake
      odds = +120  -> profit = 1.20 units on a 1u stake
    """
    odds = float(odds)
    if odds < 0:
        return 100.0 / abs(odds)
    else:
        return odds / 100.0


def model_probs_from_scores(home_score, away_score, temperature=DEFAULT_NHL_TEMPERATURE):
    """
    Convert home/away scores into probabilities via softmax with optional temperature.
    Higher temperature (>1) softens extremes; lower (<1) sharpens them.
    """
    t = max(float(temperature), 1e-6)
    eh = math.exp(home_score / t)
    ea = math.exp(away_score / t)
    total = eh + ea
    return eh / total, ea / total


def confidence_grade(edge_pct, ev_units):
    """
    Turn edge (percent) and EV (units) into a simple confidence grade.

    This is just a heuristic. You can tweak the thresholds later.
    Returns a string like "A", "B+", "C", "No Bet", etc.
    """
    if ev_units <= 0:
        return "No Bet"

    edge = abs(edge_pct)
    ev = ev_units

    # Strong edges + solid EV
    if edge >= 7.5 and ev >= 0.10:
        return "A"
    if edge >= 5.0 and ev >= 0.06:
        return "B+"
    if edge >= 3.0 and ev >= 0.03:
        return "B"
    if edge >= 1.5 and ev >= 0.01:
        return "C+"
    if edge >= 0.5 and ev > 0:
        return "C"

    return "D"  # very thin edge


def analyze_matchup_lean(home_team, away_team, games):
    """
    Balanced lean engine for a matchup (aligned with API weights).
    """
    today = date.today()
    home_profile = get_team_profile(home_team, games, today=today)
    away_profile = get_team_profile(away_team, games, today=today)
    adv_home = get_team_adv_stats(home_team) or {}
    adv_away = get_team_adv_stats(away_team) or {}

    if home_profile is None or away_profile is None:
        print("\n=== Balanced Lean Engine ===")
        print("Not enough data for one or both teams.")
        return None, None

    print("\n=== Balanced Lean Engine ===")

    def win_pct(stats):
        games = stats["games"]
        return stats["wins"] / games if games > 0 else 0.0

    home_score = 0.0
    away_score = 0.0
    reasons_home = []
    reasons_away = []

    # 1) Last 10 record (single recency factor)
    home_win10 = win_pct(home_profile["last10"])
    away_win10 = win_pct(away_profile["last10"])

    home_score += home_win10 * 1.0
    away_score += away_win10 * 1.0

    if home_win10 > away_win10 + 0.1:
        reasons_home.append("Better recent form (last 10)")
    elif away_win10 > home_win10 + 0.1:
        reasons_away.append("Better recent form (last 10)")

    # 2) Streak factor (light, capped)
    for profile, is_home_flag in ((home_profile, True), (away_profile, False)):
        streak = profile["streak"]
        length = streak["length"]
        stype = streak["type"]

        if length > 0 and stype is not None:
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

    # Flat home-ice bonus (softened)
    home_score += 0.25
    reasons_home.append("Home-ice advantage")

    # 3) Home/Road splits
    home_home = home_profile["home"]  # home team at home
    away_road = away_profile["away"]  # away team on the road

    home_home_win_pct = win_pct(home_home)
    away_road_win_pct = win_pct(away_road)

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

    # Defensive edge (lower GA in last 10) with modest weight
    home_ga10 = home_profile["last10"]["avg_against"]
    away_ga10 = away_profile["last10"]["avg_against"]

    if home_ga10 + 0.5 < away_ga10:
        home_score += 0.2
        reasons_home.append("Better defensive numbers (fewer goals against)")
    elif away_ga10 + 0.5 < home_ga10:
        away_score += 0.2
        reasons_away.append("Better defensive numbers (fewer goals against)")

    # Advanced stats edge (xG/HDCF) from MoneyPuck if available
    xgf_pct_home = adv_home.get("xgf_pct")
    xgf_pct_away = adv_away.get("xgf_pct")
    if xgf_pct_home is not None and xgf_pct_away is not None:
        delta = xgf_pct_home - xgf_pct_away
        if delta >= 3.0:
            home_score += 0.55
            reasons_home.append("Better xG share (season-to-date)")
        elif delta <= -3.0:
            away_score += 0.55
            reasons_away.append("Better xG share (season-to-date)")

    # Special teams edge (PP/PK) if meaningful
    pp_home = adv_home.get("pp")
    pk_home = adv_home.get("pk")
    pp_away = adv_away.get("pp")
    pk_away = adv_away.get("pk")
    if pp_home and pk_home and pp_away and pk_away:
        net_home = (pp_home or 0) - (100 - (pk_home or 0))
        net_away = (pp_away or 0) - (100 - (pk_away or 0))
        delta = net_home - net_away
        if delta >= 5:
            home_score += 0.15
            reasons_home.append("Special teams edge")
        elif delta <= -5:
            away_score += 0.15
            reasons_away.append("Special teams edge")

    # Rest / back-to-back
    rest_home = home_profile.get("rest") or {}
    rest_away = away_profile.get("rest") or {}
    home_days = rest_home.get("days_since_last")
    away_days = rest_away.get("days_since_last")
    if rest_home.get("is_back_to_back") and not rest_away.get("is_back_to_back"):
        home_score -= 0.40
        reasons_away.append("Home on back-to-back; away more rested")
    if rest_away.get("is_back_to_back") and not rest_home.get("is_back_to_back"):
        away_score -= 0.40
        reasons_home.append("Away on back-to-back; home more rested")
    if home_days is not None and away_days is not None:
        diff = home_days - away_days
        if diff >= 2:
            home_score += 0.20
            reasons_home.append(f"More rest ({home_days}d vs {away_days}d)")
        elif diff <= -2:
            away_score += 0.20
            reasons_away.append(f"More rest ({away_days}d vs {home_days}d)")

    # --- Decide side lean ---
    print(f"\nSide score {home_team}: {home_score:.2f}")
    print(f"Side score {away_team}: {away_score:.2f}")

    diff = home_score - away_score

    if diff > 1.5:
        print(f"\nSide Lean: {home_team} ML (strong lean)")
        if reasons_home:
            print("Reasons for home lean:")
            for r in reasons_home:
                print(f"  • {r}")
    elif diff > 0.5:
        print(f"\nSide Lean: {home_team} ML (slight lean)")
        if reasons_home:
            print("Reasons for home lean:")
            for r in reasons_home:
                print(f"  • {r}")
    elif diff < -1.5:
        print(f"\nSide Lean: {away_team} ML (strong lean)")
        if reasons_away:
            print("Reasons for away lean:")
            for r in reasons_away:
                print(f"  • {r}")
    elif diff < -0.5:
        print(f"\nSide Lean: {away_team} ML (slight lean)")
        if reasons_away:
            print("Reasons for away lean:")
            for r in reasons_away:
                print(f"  • {r}")
    else:
        print("\nSide Lean: No clear edge (scores are very close).")

    # --- TOTALS LEAN (Over/Under) ---
    home_last10 = home_profile["last10"]
    away_last10 = away_profile["last10"]

    combined_avg_for = home_last10["avg_for"] + away_last10["avg_for"]

    home_hs = home_profile["high_scoring_last5"]["count"]
    home_total_5 = home_profile["high_scoring_last5"]["total"]
    away_hs = away_profile["high_scoring_last5"]["count"]
    away_total_5 = away_profile["high_scoring_last5"]["total"]

    print(f"\nCombined L10 avg goals for: {combined_avg_for:.2f}")
    print(
        f"{home_team} 3+ goals in last 5: {home_hs}/{home_total_5}, "
        f"{away_team} 3+ goals: {away_hs}/{away_total_5}"
    )

    both_scoring_often = (home_hs >= 3 and away_hs >= 3)
    both_low_scoring = (home_hs <= 2 and away_hs <= 2)

    if combined_avg_for >= 6.5 or both_scoring_often:
        print("\nTotal Lean: OVER (based on recent scoring trends).")
    elif combined_avg_for <= 5.0 and both_low_scoring:
        print("\nTotal Lean: UNDER (based on recent lower scoring).")
    else:
        print("\nTotal Lean: Neutral (mixed scoring signals).")

    # Return scores so we can convert them into probabilities for EV.
    return home_score, away_score


def run_betting_edge_section(home_team, away_team, home_score, away_score):
    """
    Compute the expected value (EV) for a 1-unit bet on each side.

    First, we TRY to auto-load odds from ESPN for today's games.
    If that fails, we fall back to asking the user for moneyline odds.

    This uses:
      - Model probabilities from our lean scores
      - Market implied probabilities from the odds
    """
    if home_score is None or away_score is None:
        print("\n[EV] Skipping EV calculation (no scores from lean engine).")
        return

    print("\n=== Betting Edge (Moneyline EV) ===")
    print("We'll try to auto-load ESPN odds for today's matchup.")
    print("If that fails, you'll be asked to enter odds manually.\n")

    home_odds = None
    away_odds = None

    # 1) Try to auto-load odds from ESPN
    auto_odds = load_current_odds_for_matchup(home_team, away_team)
    if auto_odds is not None:
        home_odds = auto_odds.get("home_ml")
        away_odds = auto_odds.get("away_ml")

        if home_odds is not None and away_odds is not None:
            print(
                f"Using ESPN odds for today:\n"
                f"  {home_team} moneyline: {home_odds:+.0f}\n"
                f"  {away_team} moneyline: {away_odds:+.0f}"
            )
        else:
            home_odds = None
            away_odds = None

    # 2) If auto-odds failed, ask the user
    if home_odds is None or away_odds is None:
        print("\nCould not auto-load odds for this matchup.")
        print("Please enter American-style moneyline odds (e.g. -135, +120).")

        try:
            home_odds_str = input(f"Enter HOME ({home_team}) moneyline odds: ").strip()
            away_odds_str = input(f"Enter AWAY ({away_team}) moneyline odds: ").strip()

            home_odds = float(home_odds_str)
            away_odds = float(away_odds_str)
        except ValueError:
            print("Invalid odds entered. Skipping EV calculation.")
            return

    # Model probabilities from scores
    p_home_model, p_away_model = model_probs_from_scores(
        home_score, away_score, temperature=DEFAULT_NHL_TEMPERATURE
    )

    # Market implied probabilities from odds
    p_home_market = implied_prob_american(home_odds)
    p_away_market = implied_prob_american(away_odds)

    # Profit on a 1-unit stake if bet wins
    home_profit_on_win = profit_on_win_for_1_unit(home_odds)
    away_profit_on_win = profit_on_win_for_1_unit(away_odds)

    # EV = p(win)*profit - p(loss)*1
    home_ev = p_home_model * home_profit_on_win - (1 - p_home_model) * 1.0
    away_ev = p_away_model * away_profit_on_win - (1 - p_away_model) * 1.0

    # Edges vs market
    home_edge_pct = (p_home_model - p_home_market) * 100.0
    away_edge_pct = (p_away_model - p_away_market) * 100.0
    home_is_favorite = p_home_market >= p_away_market
    away_is_favorite = p_away_market >= p_home_market
    home_edge_required = (
        MIN_EDGE_FAVORITE if home_is_favorite else MIN_EDGE_UNDERDOG
    ) * 100.0
    away_edge_required = (
        MIN_EDGE_FAVORITE if away_is_favorite else MIN_EDGE_UNDERDOG
    ) * 100.0
    home_meets_filters = (
        p_home_model >= MIN_MODEL_PROBABILITY and home_edge_pct >= home_edge_required
    )
    away_meets_filters = (
        p_away_model >= MIN_MODEL_PROBABILITY and away_edge_pct >= away_edge_required
    )

    print("\n--- Model vs Market ---")
    print(
        f"{home_team}: Model prob = {p_home_model*100:.1f}% | "
        f"Market implied = {p_home_market*100:.1f}% | "
        f"Edge = {home_edge_pct:+.1f}%"
    )
    print(
        f"{away_team}: Model prob = {p_away_model*100:.1f}% | "
        f"Market implied = {p_away_market*100:.1f}% | "
        f"Edge = {away_edge_pct:+.1f}%"
    )

    print("\n--- EV for 1-unit stake ---")
    print(
        f"If you bet 1 unit on {home_team}: "
        f"EV = {home_ev:+.3f} units (profit on win = {home_profit_on_win:.3f})"
    )
    print(
        f"If you bet 1 unit on {away_team}: "
        f"EV = {away_ev:+.3f} units (profit on win = {away_profit_on_win:.3f})"
    )

    # Confidence grades
    home_grade = confidence_grade(home_edge_pct, home_ev)
    away_grade = confidence_grade(away_edge_pct, away_ev)

    print("\n--- Confidence Grades ---")
    print(f"{home_team} ML: grade {home_grade}")
    print(f"{away_team} ML: grade {away_grade}")

    # Simple recommendation based on EV (unchanged logic)
    print("\n--- Simple EV Lean ---")
    if home_ev > 0 and home_ev > away_ev and home_meets_filters:
        print(
            f"Model shows +EV on {home_team} moneyline "
            f"(best EV of the two sides)."
        )
    elif away_ev > 0 and away_ev > home_ev and away_meets_filters:
        print(
            f"Model shows +EV on {away_team} moneyline "
            f"(best EV of the two sides)."
        )
    elif home_ev <= 0 and away_ev <= 0:
        print("Model does NOT see a clear +EV edge on either moneyline side.")
    elif not home_meets_filters and not away_meets_filters:
        print(
            "Model EV exists but neither side meets the probability/edge filters "
            "(fav edge >= 5%, dog edge >= 22%)."
        )
    else:
        print("EV is mixed. Neither side is clearly dominant by EV alone.")

    # --- CLEAN SUMMARY LINE (for quick use/posting) ---
    print("\n=== PICK SUMMARY ===")

    # Decide best side by EV
    best_team = None
    best_ev = None
    best_edge = None
    best_grade = None
    best_model_prob = None
    best_market_prob = None
    best_odds = None

    if (home_ev > 0 and home_meets_filters) or (away_ev > 0 and away_meets_filters):
        if home_ev > away_ev and home_meets_filters:
            best_team = home_team
            best_ev = home_ev
            best_edge = home_edge_pct
            best_grade = home_grade
            best_model_prob = p_home_model
            best_market_prob = p_home_market
            best_odds = home_odds
        elif away_meets_filters:
            best_team = away_team
            best_ev = away_ev
            best_edge = away_edge_pct
            best_grade = away_grade
            best_model_prob = p_away_model
            best_market_prob = p_away_market
            best_odds = away_odds

    if best_team is None or best_grade == "No Bet":
        print("No clear +EV pick. (Both sides graded as No Bet or EV <= 0.)")
    else:
        print(
            f"Pick: {best_team} ML "
            f"| Odds: {best_odds:+.0f} "
            f"| Model: {best_model_prob*100:.1f}% vs Market: {best_market_prob*100:.1f}% "
            f"| Edge: {best_edge:+.1f}% "
            f"| EV: {best_ev:+.3f}u "
            f"| Confidence: {best_grade}"
        )


def simple_last5_lean(home_team, away_team, games):
    """
    Simple lean based only on last 5 games for each team.
    """
    home_games = get_team_games(games, home_team)
    away_games = get_team_games(games, away_team)

    home_last = calculate_last_n_stats(home_team, home_games, n=5)
    away_last = calculate_last_n_stats(away_team, away_games, n=5)

    print("\n=== Simple Lean (Last 5 Games) ===")

    # Side lean: who has more wins in last 5?
    if home_last["wins"] > away_last["wins"]:
        print(
            f"Side lean: {home_team} "
            f"(last-5 wins {home_last['wins']} vs {away_last['wins']})."
        )
    elif away_last["wins"] > home_last["wins"]:
        print(
            f"Side lean: {away_team} "
            f"(last-5 wins {away_last['wins']} vs {home_last['wins']})."
        )
    else:
        print("Side lean: No clear edge on wins in last 5 (tied).")

    combined_avg_goals = home_last["avg_for"] + away_last["avg_for"]
    print(f"Combined avg goals for (last 5): {combined_avg_goals:.2f}")

    if combined_avg_goals >= 6:
        print(
            "Total lean: Higher scoring trend "
            "(potential OVER spot based on recent form)."
        )
    elif combined_avg_goals <= 5:
        print(
            "Total lean: Lower scoring trend "
            "(potential UNDER spot based on recent form)."
        )
    else:
        print("Total lean: Neutral, recent scoring is mid-range.")


# =========================
#   FUTURE MATCHUP MODE
# =========================

def run_future_matchup_mode(games):
    """
    Show the schedule for a future date (default: tomorrow),
    let the user pick a matchup, then run:
      - Team analysis (home + away)
      - Balanced lean
      - EV
      - Simple last-5 lean
    """
    print("\n=== Future Matchup Mode ===")

    tomorrow = date.today() + timedelta(days=1)
    default_date_str = tomorrow.isoformat()

    user_input = input(
        f"Enter date for schedule (YYYY-MM-DD) or press Enter for {default_date_str}: "
    ).strip()

    if user_input:
        target_date = user_input
    else:
        target_date = default_date_str

    schedule = load_schedule_for_date(target_date)

    if not schedule:
        print(f"\nNo games found on {target_date}.")
        return

    print(f"\nGames on {target_date}:")
    for idx, g in enumerate(schedule, start=1):
        print(f"{idx}) {g['away_team']} @ {g['home_team']}")

    choice_str = input(
        "\nEnter the number of the matchup you want to analyze: "
    ).strip()

    try:
        choice = int(choice_str)
    except ValueError:
        print("Invalid choice (not a number).")
        return

    if choice < 1 or choice > len(schedule):
        print("Choice out of range.")
        return

    selected = schedule[choice - 1]
    home_team = selected["home_team"]
    away_team = selected["away_team"]

    print(f"\nYou selected: {away_team} @ {home_team} on {target_date}")

    print("\n--- Home Team ---")
    analyze_team(home_team, games)

    print("\n--- Away Team ---")
    analyze_team(away_team, games)

    # Balanced lean engine
    home_score, away_score = analyze_matchup_lean(home_team, away_team, games)

    # Betting Edge (EV)
    run_betting_edge_section(home_team, away_team, home_score, away_score)

    # Simple last-5 lean
    simple_last5_lean(home_team, away_team, games)


# =========================
#   MAIN PROGRAM FLOW
# =========================

def main():
    print("NHL Betting Analyzer")
    print("====================")

    print("\nHow do you want to load data?")
    print("1) From ESPN (custom date range, e.g. full season)")
    print("2) From ESPN (today's completed games only)")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        start_date_str = input(
            "\nEnter START date (YYYY-MM-DD), e.g. 2024-10-01: "
        ).strip()
        end_date_str = input(
            "Enter END date (YYYY-MM-DD), e.g. 2025-04-20: "
        ).strip()

        try:
            games = load_games_from_espn_date_range(start_date_str, end_date_str)
        except Exception as e:
            print("\n[API ERROR] Something went wrong loading date range from ESPN.")
            print("Details:", e)
            return

    elif choice == "2":
        print("\nLoading today's completed games from ESPN NHL scoreboard...")
        try:
            games = load_games_from_espn_scoreboard()
        except Exception as e:
            print("\n[API ERROR] Something went wrong while calling ESPN.")
            print("Details:", e)
            return
    else:
        print("Invalid choice.")
        return

    print(f"\nTotal games loaded: {len(games)}\n")

    if len(games) == 0:
        print("No games available in the dataset. Cannot run analysis.")
        return

    # Print each game once for context
    for game in games:
        line = (
            f"{game['date']}: "
            f"{game['away_team']} @ {game['home_team']} "
            f"({game['away_goals']} - {game['home_goals']})"
        )
        print(line)

    # ===== MENU LOOP =====
    while True:
        print("\n========================")
        print("What would you like to do?")
        print("1) Analyze a single team")
        print("2) Analyze a matchup (with leans + EV)")
        print("3) Future Matchup Mode (pick from schedule)")
        print("4) Exit")
        print("========================")

        menu_choice = input("Enter 1, 2, 3, or 4: ").strip()

        if menu_choice == "1":
            # Single team analysis
            team = input(
                "\nEnter team code for analysis (e.g. TOR, EDM, BOS): "
            ).strip().upper()
            analyze_team(team, games)

        elif menu_choice == "2":
            # Full matchup workflow: team analysis + lean + EV + simple last-5 lean
            print("\n=== Matchup Analysis ===")
            home_team = input("Enter HOME team code: ").strip().upper()
            away_team = input("Enter AWAY team code: ").strip().upper()

            print("\n--- Home Team ---")
            analyze_team(home_team, games)

            print("\n--- Away Team ---")
            analyze_team(away_team, games)

            # Balanced lean engine
            home_score, away_score = analyze_matchup_lean(home_team, away_team, games)

            # Betting Edge (EV)
            run_betting_edge_section(home_team, away_team, home_score, away_score)

            # Simple last-5 lean
            simple_last5_lean(home_team, away_team, games)

        elif menu_choice == "3":
            # Future Matchup Mode
            run_future_matchup_mode(games)

        elif menu_choice == "4":
            print("\nExiting analyzer. Good luck with your bets!")
            break

        else:
            print("\nInvalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
