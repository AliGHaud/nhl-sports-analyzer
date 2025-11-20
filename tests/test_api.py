import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import api


client = TestClient(api.app)


def sample_games():
    return [
        {
            "date": "2025-11-01",
            "home_team": "BOS",
            "away_team": "TOR",
            "home_goals": "3",
            "away_goals": "2",
        },
        {
            "date": "2025-11-03",
            "home_team": "TOR",
            "away_team": "BOS",
            "home_goals": "1",
            "away_goals": "4",
        },
    ]


def test_matchup_basic_no_odds():
    with patch("api.load_games_from_espn_date_range", return_value=sample_games()), patch(
        "api.load_current_odds_for_matchup", return_value=None
    ):
        resp = client.get("/nhl/matchup", params={"home": "BOS", "away": "TOR"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["ev"] is None
    assert "lean" in data["side"]
    assert data["profiles"]["home"]["team"] == "BOS"
    assert data["profiles"]["away"]["team"] == "TOR"


def test_matchup_with_odds():
    odds = {"home_ml": -120, "away_ml": 110}
    with patch("api.load_games_from_espn_date_range", return_value=sample_games()), patch(
        "api.load_current_odds_for_matchup", return_value=odds
    ):
        resp = client.get("/nhl/matchup", params={"home": "BOS", "away": "TOR"})
    assert resp.status_code == 200
    ev = resp.json()["ev"]
    assert ev is not None
    assert "ev_units" in ev
    assert "grades" in ev


def test_matchup_bad_date():
    resp = client.get("/nhl/matchup", params={"home": "BOS", "away": "TOR", "start_date": "bad"})
    assert resp.status_code == 400


def test_matchup_unknown_team():
    resp = client.get("/nhl/matchup", params={"home": "XXX", "away": "TOR"})
    assert resp.status_code == 400


def test_team_endpoint():
    with patch("api.load_games_from_espn_date_range", return_value=sample_games()):
        resp = client.get("/nhl/team", params={"team": "BOS"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["profile"]["team"] == "BOS"
    assert "last5" in body["profile"]
    assert "last10" in body["profile"]


def test_today_schedule():
    fake_schedule = [
        {"home_team": "BOS", "away_team": "TOR"},
        {"home_team": "NYR", "away_team": "MTL"},
    ]
    with patch("api.load_schedule_for_date", return_value=fake_schedule), patch(
        "api.load_current_odds_for_matchup", return_value={"home_ml": -120, "away_ml": 110}
    ):
        resp = client.get("/nhl/today")
    assert resp.status_code == 200
    games = resp.json()["games"]
    assert len(games) == 2
    assert games[0]["home"] == "BOS"
    assert games[0]["odds"]["home_ml"] == -120


def test_pick_endpoint_with_candidate():
    schedule = [{"home_team": "BOS", "away_team": "TOR"}]
    reasons = {"home_reasons": ["solid"], "away_reasons": []}
    ev_block = {
        "odds": {"home_ml": -120, "away_ml": 110},
        "model_prob": {"home": 0.55, "away": 0.45},
        "market_prob": {"home": 0.52, "away": 0.48},
        "edge_pct": {"home": 3.0, "away": -3.0},
        "ev_units": {"home": 0.05, "away": -0.05},
        "grades": {"home": "B", "away": "D"},
    }
    with patch("api.load_schedule_for_date", return_value=schedule), patch(
        "api.load_games_from_espn_date_range", return_value=sample_games()
    ), patch("api._lean_scores", return_value=(1.0, 0.5, reasons)), patch(
        "api._ev_block", return_value=ev_block
    ):
        resp = client.get("/nhl/pick")
    assert resp.status_code == 200
    body = resp.json()
    assert body["pick"] is not None
    assert body["pick"]["side"] == "BOS"


def test_pick_endpoint_no_candidate():
    schedule = [{"home_team": "BOS", "away_team": "TOR"}]
    reasons = {"home_reasons": [], "away_reasons": []}
    ev_block = {
        "odds": {"home_ml": -120, "away_ml": 110},
        "model_prob": {"home": 0.51, "away": 0.49},
        "market_prob": {"home": 0.50, "away": 0.50},
        "edge_pct": {"home": 1.0, "away": -1.0},
        "ev_units": {"home": 0.01, "away": -0.01},
        "grades": {"home": "C", "away": "D"},
    }
    with patch("api.load_schedule_for_date", return_value=schedule), patch(
        "api.load_games_from_espn_date_range", return_value=sample_games()
    ), patch("api._lean_scores", return_value=(1.0, 0.5, reasons)), patch(
        "api._ev_block", return_value=ev_block
    ):
        resp = client.get("/nhl/pick")
    assert resp.status_code == 200
    body = resp.json()
    assert body["pick"] is None
    assert "No qualifying pick" in body["reason"]


def test_sports_endpoint():
    resp = client.get("/sports")
    assert resp.status_code == 200
    data = resp.json()
    assert "sports" in data
    assert any(s["id"] == "nhl" for s in data["sports"])
