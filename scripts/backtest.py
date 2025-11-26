"""
Simple backtest harness for the NHL lean engine.

Usage (default full season window):
  python scripts/backtest.py

Custom dates:
  python scripts/backtest.py --start 2024-10-01 --end 2025-04-15
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

from datetime import datetime, timedelta

from data_sources import load_games_from_espn_date_range
from api import _lean_scores
from nhl_analyzer import model_probs_from_scores

# Local helper for odds data
SCRIPT_DIR = Path(__file__).resolve().parent
import sys

if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))
from odds_loader import load_clean_odds  # noqa: E402


def log_loss(prob, outcome):
    """Binary log loss."""
    # Clamp to avoid log(0)
    p = min(max(prob, 1e-9), 1 - 1e-9)
    return -math.log(p if outcome == 1 else (1 - p))


def brier(prob, outcome):
    return (prob - outcome) ** 2


def bucket(prob, width=0.1):
    """Return bucket string like '0.5-0.6'."""
    lo = math.floor(prob / width) * width
    hi = lo + width
    return f"{lo:.1f}-{hi:.1f}"


def american_to_prob(us: float) -> float:
    if us is None:
        raise ValueError("missing odds value")
    if us > 0:
        return 100.0 / (us + 100.0)
    return -us / (-us + 100.0)


def american_to_decimal(us: float) -> float:
    if us > 0:
        return us / 100.0 + 1.0
    return 100.0 / (-us) + 1.0


def backtest(
    start_date: str,
    end_date: str,
    odds_file: str | None = None,
    min_edge: float = 0.05,
    min_prob: float = 0.5,
    stake: float = 1.0,
):
    games = load_games_from_espn_date_range(start_date, end_date, force_refresh=False)
    if not games:
        print("No games loaded for range", start_date, end_date)
        return

    total = 0
    ll_sum = 0.0
    brier_sum = 0.0
    correct = 0
    buckets = defaultdict(lambda: {"n": 0, "wins": 0})
    failures = 0
    odds_hits = 0
    edge_sum = 0.0
    bet_count = 0
    bet_wins = 0
    total_profit = 0.0
    total_staked = 0.0
    fav_split = {
        "favorite": {"bets": 0, "wins": 0, "profit": 0.0},
        "underdog": {"bets": 0, "wins": 0, "profit": 0.0},
    }
    side_split = {
        "home": {"bets": 0, "wins": 0, "profit": 0.0},
        "away": {"bets": 0, "wins": 0, "profit": 0.0},
    }

    odds_lookup = {}
    if odds_file:
        odds_lookup = load_clean_odds(odds_file)

    prob_threshold = max(min_prob, 0.5)

    for g in games:
        try:
            home = g["home_team"]
            away = g["away_team"]
            hg = int(g["home_goals"])
            ag = int(g["away_goals"])
        except Exception:
            continue

        outcome = 1 if hg > ag else 0  # home win = 1

        try:
            home_score, away_score, _ = _lean_scores(
                home,
                away,
                games,
                force_refresh=False,
                injuries_home=None,
                injuries_away=None,
                projected_goalies=None,
            )
        except Exception:
            failures += 1
            continue

        prob_home, prob_away = model_probs_from_scores(home_score, away_score)
        total += 1
        correct += 1 if (prob_home >= 0.5 and outcome == 1) or (prob_home < 0.5 and outcome == 0) else 0
        ll_sum += log_loss(prob_home, outcome)
        brier_sum += brier(prob_home, outcome)

        b = bucket(prob_home)
        buckets[b]["n"] += 1
        buckets[b]["wins"] += outcome

        if odds_lookup:
            # Try exact date first, then ±1 day for timezone mismatches
            game_date = datetime.strptime(g["date"], "%Y-%m-%d")
            day_before = (game_date - timedelta(days=1)).strftime("%Y-%m-%d")
            day_after = (game_date + timedelta(days=1)).strftime("%Y-%m-%d")

            odds = None
            for date_key in [g["date"], day_before, day_after]:
                key = (date_key, home, away)
                if key in odds_lookup:
                    odds = odds_lookup[key]
                    break

            if odds and odds.get("home_ml_close") and odds.get("away_ml_close"):
                odds_hits += 1
                home_ml = float(odds["home_ml_close"])
                away_ml = float(odds["away_ml_close"])
                implied_home = american_to_prob(home_ml)
                implied_away = american_to_prob(away_ml)

                home_edge = prob_home - implied_home
                away_edge = prob_away - implied_away

                bet = None
                if (
                    prob_home >= prob_threshold
                    and home_edge > min_edge
                    and home_edge > away_edge
                ):
                    bet = {
                        "side": "home",
                        "edge": home_edge,
                        "prob": prob_home,
                        "implied": implied_home,
                        "line": home_ml,
                        "win": outcome == 1,
                        "other_implied": implied_away,
                    }
                elif (
                    prob_away >= prob_threshold
                    and away_edge > min_edge
                    and away_edge > home_edge
                ):
                    bet = {
                        "side": "away",
                        "edge": away_edge,
                        "prob": prob_away,
                        "implied": implied_away,
                        "line": away_ml,
                        "win": outcome == 0,
                        "other_implied": implied_home,
                    }

                if bet:
                    bet_count += 1
                    edge_sum += bet["edge"]
                    total_staked += stake
                    decimal_odds = american_to_decimal(bet["line"])
                    if bet["win"]:
                        bet_wins += 1
                        profit = stake * (decimal_odds - 1.0)
                    else:
                        profit = -stake
                    total_profit += profit
                    fav_key = (
                        "favorite"
                        if bet["implied"] >= bet["other_implied"]
                        else "underdog"
                    )
                    fav_split[fav_key]["bets"] += 1
                    fav_split[fav_key]["profit"] += profit
                    if bet["win"]:
                        fav_split[fav_key]["wins"] += 1
                    side_key = bet["side"]
                    side_split[side_key]["bets"] += 1
                    side_split[side_key]["profit"] += profit
                    if bet["win"]:
                        side_split[side_key]["wins"] += 1

    if total == 0:
        print("No usable games found.")
        return

    print(f"Backtest {start_date} to {end_date}")
    print(f"Games evaluated: {total} (failures: {failures})")
    print(f"Accuracy: {correct/total*100:.2f}%")
    print(f"Avg log loss: {ll_sum/total:.4f}")
    print(f"Avg Brier score: {brier_sum/total:.4f}")
    print("\nCalibration (home prob buckets):")
    for k in sorted(buckets.keys()):
        n = buckets[k]["n"]
        if n == 0:
            continue
        win_rate = buckets[k]["wins"] / n
        print(f"  {k}: n={n}, actual_home_win={win_rate*100:.1f}%")

    if odds_lookup:
        print(f"\nOdds coverage: {odds_hits}/{total}")
        print(
            f"Bet filter -> stake: {stake:.2f}u, min_prob: {prob_threshold:.2f}, "
            f"min_edge: {min_edge*100:.1f}%"
        )
        if bet_count:
            print(f"Bets meeting filter: {bet_count}")
            avg_edge = edge_sum / bet_count * 100.0
            hit_rate = bet_wins / bet_count * 100.0
            roi = total_profit / total_staked * 100.0 if total_staked else 0.0
            print(
                f"Avg edge: {avg_edge:.2f}% | Hit rate: {hit_rate:.2f}% | "
                f"Profit: {total_profit:.2f}u on {total_staked:.2f}u staked "
                f"(ROI {roi:.2f}%)"
            )
            print("Favorite vs. Underdog results:")
            for label, stats in fav_split.items():
                if not stats["bets"]:
                    continue
                fav_roi = (
                    stats["profit"] / (stats["bets"] * stake) * 100.0 if stake else 0.0
                )
                fav_hit = stats["wins"] / stats["bets"] * 100.0
                print(
                    f"  {label.capitalize()}: {stats['bets']} bets, "
                    f"hit {fav_hit:.1f}%, profit {stats['profit']:.2f}u "
                    f"(ROI {fav_roi:.2f}%)"
                )
            print("Home vs. Away results:")
            for label, stats in side_split.items():
                if not stats["bets"]:
                    continue
                side_roi = (
                    stats["profit"] / (stats["bets"] * stake) * 100.0 if stake else 0.0
                )
                side_hit = stats["wins"] / stats["bets"] * 100.0
                print(
                    f"  {label.capitalize()}: {stats['bets']} bets, "
                    f"hit {side_hit:.1f}%, profit {stats['profit']:.2f}u "
                    f"(ROI {side_roi:.2f}%)"
                )
        else:
            print("No games met the betting filter.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest NHL lean engine.")
    parser.add_argument("--start", default="2024-10-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2025-04-15", help="End date YYYY-MM-DD")
    parser.add_argument("--odds", help="Path to clean odds CSV (optional)")
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.05,
        help="Minimum model edge (prob - market) required to count a bet (default 0.05)",
    )
    parser.add_argument(
        "--min-prob",
        type=float,
        default=0.5,
        help="Minimum model probability threshold required to count a bet (default 0.5)",
    )
    parser.add_argument(
        "--stake",
        type=float,
        default=1.0,
        help="Flat stake (units) for each qualifying bet (default 1.0)",
    )
    args = parser.parse_args()
    backtest(
        args.start,
        args.end,
        odds_file=args.odds,
        min_edge=args.min_edge,
        min_prob=args.min_prob,
        stake=args.stake,
    )
