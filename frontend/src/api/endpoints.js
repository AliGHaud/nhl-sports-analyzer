const API_BASE = import.meta.env.VITE_API_BASE || (typeof window !== 'undefined' ? window.location.origin : 'http://localhost:8000')

export const endpoints = {
  nhl: {
    moneyline: `${API_BASE}/nhl/picks`,
    spreads: `${API_BASE}/nhl/picks`,
    schedule: `${API_BASE}/nhl/today`,
    matchup: `${API_BASE}/nhl/matchup`,
  },
  nfl: {
    moneyline: `${API_BASE}/nfl/picks`,
    spreads: `${API_BASE}/nfl/spread-picks`,
    schedule: `${API_BASE}/nfl/today`,
    matchup: `${API_BASE}/nfl/matchup`,
  },
}
