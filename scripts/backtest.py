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
import json
from pathlib import Path

from datetime import datetime, timedelta
from typing import Dict, Any, List

from tqdm import tqdm

from data_sources import load_games_from_espn_date_range
from api import _lean_scores
from nhl_analyzer import model_probs_from_scores

# Local helper for odds data
SCRIPT_DIR = Path(__file__).resolve().parent
import sys

if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))
from odds_loader import load_clean_odds  # noqa: E402


def validate_backtest_input(odds_data: Dict, start_date: str, end_date: str) -> None:
    """Basic validation that the odds file has content and required fields."""
    if odds_data is None:
        raise ValueError("Odds data is None; cannot validate.")
    if not odds_data:
        raise ValueError("Empty odds file; no rows found.")
    sample = next(iter(odds_data.values()))
    required_keys = {"home_ml_close", "away_ml_close", "date", "home_team", "away_team"}
    missing = [k for k in required_keys if k not in sample]
    if missing:
        raise ValueError(f"Odds rows missing required columns: {missing}")
    print(
        f"Validated odds file with {len(odds_data)} rows "
        f"for range {start_date} to {end_date}."
    )


def calibration_bucket_label(prob: float, step: float = 0.05) -> str:
    """Bucket probabilities into strings like '0.50-0.55'."""
    lo = max(0.0, math.floor(prob / step) * step)
    hi = min(1.0, lo + step)
    return f"{lo:.2f}-{hi:.2f}"


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
    quiet: bool = False,
) -> Dict[str, Any]:
    games = load_games_from_espn_date_range(start_date, end_date, force_refresh=False)
    if not games:
        print("No games loaded for range", start_date, end_date)
        return {}

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
    bet_calibration: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"bets": 0, "wins": 0}
    )

    if odds_file:
        odds_lookup = load_clean_odds(odds_file)
        validate_backtest_input(odds_lookup, start_date, end_date)

    prob_threshold = max(min_prob, 0.5)

    iterator = games
    if not quiet:
        iterator = tqdm(games, desc="Backtesting", unit="game")

    for g in iterator:
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
                    bucket_key = calibration_bucket_label(bet["prob"])
                    bet_calibration[bucket_key]["bets"] += 1
                    if bet["win"]:
                        bet_calibration[bucket_key]["wins"] += 1

    if total == 0:
        print("No usable games found.")
        return {}

    summary: Dict[str, Any] = {
        "start_date": start_date,
        "end_date": end_date,
        "total_games": total,
        "failures": failures,
        "accuracy": correct / total if total else 0.0,
        "avg_log_loss": ll_sum / total if total else None,
        "avg_brier": brier_sum / total if total else None,
        "odds_coverage": odds_hits,
        "bet_filters": {
            "stake": stake,
            "min_prob": prob_threshold,
            "min_edge": min_edge,
        },
        "bet_results": {
            "bets": bet_count,
            "wins": bet_wins,
            "avg_edge": edge_sum / bet_count if bet_count else 0.0,
            "profit_units": total_profit,
            "total_staked": total_staked,
            "roi": (total_profit / total_staked) if total_staked else 0.0,
        },
        "favorite_split": fav_split,
        "side_split": side_split,
        "calibration": {k: v for k, v in buckets.items()},
        "bet_calibration": {
            k: {
                "bets": v["bets"],
                "wins": v["wins"],
                "hit_rate": (v["wins"] / v["bets"]) if v["bets"] else None,
            }
            for k, v in sorted(bet_calibration.items())
        },
    }

    results_path = f"backtest_results_{start_date}_to_{end_date}_edge_{min_edge:.3f}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if not quiet:
        print(f"\nSaved results to {results_path}")

    if not quiet:
        print(f"\nBacktest {start_date} to {end_date}")
        print(f"Games evaluated: {total} (failures: {failures})")
        print(f"Accuracy: {summary['accuracy']*100:.2f}%")
        print(f"Avg log loss: {summary['avg_log_loss']:.4f}")
        print(f"Avg Brier score: {summary['avg_brier']:.4f}")
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
                avg_edge_pct = summary["bet_results"]["avg_edge"] * 100.0
                hit_rate = bet_wins / bet_count * 100.0
                roi_pct = summary["bet_results"]["roi"] * 100.0
                print(f"Bets meeting filter: {bet_count}")
                print(
                    f"Avg edge: {avg_edge_pct:.2f}% | Hit rate: {hit_rate:.2f}% | "
                    f"Profit: {total_profit:.2f}u on {total_staked:.2f}u staked "
                    f"(ROI {roi_pct:.2f}%)"
                )
                print("Favorite vs. Underdog results:")
                for label, stats in fav_split.items():
                    if not stats["bets"]:
                        continue
                    fav_roi = (
                        stats["profit"] / (stats["bets"] * stake) * 100.0
                        if stake
                        else 0.0
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
                        stats["profit"] / (stats["bets"] * stake) * 100.0
                        if stake
                        else 0.0
                    )
                    side_hit = stats["wins"] / stats["bets"] * 100.0
                    print(
                        f"  {label.capitalize()}: {stats['bets']} bets, "
                        f"hit {side_hit:.1f}%, profit {stats['profit']:.2f}u "
                        f"(ROI {side_roi:.2f}%)"
                    )
                if bet_calibration:
                    print("\nBet calibration buckets:")
                    for label, stats in summary["bet_calibration"].items():
                        if not stats["bets"]:
                            continue
                        print(
                            f"  {label}: bets={stats['bets']}, "
                            f"hit={stats['hit_rate']*100:.1f}%"
                        )
            else:
                print("No games met the betting filter.")

    return summary


def run_edge_sweep(
    start_date: str,
    end_date: str,
    odds_file: str,
    edges: List[float],
    min_prob: float,
    stake: float,
) -> List[Dict[str, Any]]:
    print(f"\nRunning edge sweep for {edges}")
    results = []
    for edge in edges:
        result = backtest(
            start_date,
            end_date,
            odds_file=odds_file,
            min_edge=edge,
            min_prob=min_prob,
            stake=stake,
            quiet=True,
        )
        if result:
            results.append(result)
            roi_pct = result["bet_results"]["roi"] * 100.0
            print(
                f"Edge {edge:.3f}: ROI={roi_pct:.2f}% | Bets={result['bet_results']['bets']} "
                f"| Profit={result['bet_results']['profit_units']:.2f}u"
            )
        else:
            print(f"Edge {edge:.3f}: No results")
    return results


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
    parser.add_argument(
        "--edge-sweep",
        help="Comma-separated list of edge thresholds to evaluate (e.g., 0.01,0.03,0.05).",
    )
    args = parser.parse_args()
    if args.edge_sweep:
        sweep_values = []
        for chunk in args.edge_sweep.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                sweep_values.append(float(chunk))
            except ValueError:
                print(f"Skipping invalid edge value '{chunk}'")
        if not sweep_values:
            print("No valid edge values supplied for sweep. Running default backtest.")
            backtest(
                args.start,
                args.end,
                odds_file=args.odds,
                min_edge=args.min_edge,
                min_prob=args.min_prob,
                stake=args.stake,
            )
        else:
            run_edge_sweep(
                args.start,
                args.end,
                odds_file=args.odds,
                edges=sweep_values,
                min_prob=args.min_prob,
                stake=args.stake,
            )
    else:
        backtest(
            args.start,
            args.end,
            odds_file=args.odds,
            min_edge=args.min_edge,
            min_prob=args.min_prob,
            stake=args.stake,
        )
