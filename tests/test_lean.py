import math

import pytest

from nhl_analyzer import model_probs_from_scores
from api import _pick_candidate_from_ev


def test_model_probs_temperature_softens_extremes():
    # Without temperature, large score gap -> extreme prob
    p_hot = model_probs_from_scores(4.0, -4.0, temperature=1.0)[0]
    # With temperature > 1, prob should move closer to 0.5
    p_soft = model_probs_from_scores(4.0, -4.0, temperature=1.2)[0]
    assert p_hot > 0.97
    assert 0.6 < p_soft < p_hot


def test_pick_filters_require_edge_ev_model_prob_and_odds_cap():
    reasons = {"goalie": {"error": None, "home": {"start_prob": 1.0}, "away": {"start_prob": 1.0}}}
    ev_block = {
        "ev_units": {"home": 0.06, "away": 0.06},
        "edge_pct": {"home": 3.5, "away": 3.5},
        "model_prob": {"home": 0.58, "away": 0.42},
        "market_prob": {"home": 0.53, "away": 0.47},
        "odds": {"home_ml": -140, "away_ml": 130},
        "grades": {"home": "B", "away": "B"},
    }
    # Slate size small -> looser thresholds, should pass for home
    cand = _pick_candidate_from_ev("HOME", "AWAY", ev_block, reasons, slate_size=2)
    assert cand is not None
    assert cand["side"] == "HOME"

    # Too little edge/EV should fail
    ev_block["ev_units"]["home"] = 0.01
    ev_block["edge_pct"]["home"] = 0.5
    cand2 = _pick_candidate_from_ev("HOME", "AWAY", ev_block, reasons, slate_size=10)
    assert cand2 is None

    # Odds outside cap should fail
    ev_block["ev_units"]["home"] = 0.06
    ev_block["edge_pct"]["home"] = 3.5
    ev_block["odds"]["home_ml"] = -200
    cand3 = _pick_candidate_from_ev("HOME", "AWAY", ev_block, reasons, slate_size=2)
    assert cand3 is None

    # Goalie uncertainty should fail
    ev_block["odds"]["home_ml"] = -140
    reasons["goalie"]["home"]["start_prob"] = 0.4
    cand4 = _pick_candidate_from_ev("HOME", "AWAY", ev_block, reasons, slate_size=2)
    assert cand4 is None
