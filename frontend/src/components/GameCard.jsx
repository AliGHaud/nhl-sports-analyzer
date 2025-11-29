import { useState } from 'react'
import { endpoints } from '../api/endpoints'

export default function GameCard({ sport, game, onAnalysis }) {
  const [analysis, setAnalysis] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const home = game?.home_team || game?.home || 'HOME'
  const away = game?.away_team || game?.away || 'AWAY'

  const analyze = async () => {
    if (analysis) {
      setAnalysis(null)
      setError(null)
      return
    }
    setLoading(true)
    setError(null)
    setAnalysis(null)
    try {
      const endpoint = endpoints?.[sport]?.matchup
      if (!endpoint) throw new Error('Matchup endpoint missing')
      const url = `${endpoint}?home=${encodeURIComponent(home)}&away=${encodeURIComponent(away)}`
      const res = await fetch(url)
      let data = null
      try {
        data = await res.json()
      } catch (_) {
        /* ignore */
      }
      if (!res.ok) {
        const msg = data?.detail || `HTTP ${res.status}`
        throw new Error(msg)
      }
      setAnalysis(data)
    } catch (err) {
      setError(err.message || 'Failed to analyze')
    } finally {
      setLoading(false)
    }
  }

  const oddsText = (() => {
    const o = game?.odds
    if (o && (o.home_ml || o.away_ml)) {
      return `${o.away_ml ?? ''} / ${o.home_ml ?? ''}`
    }
    if (o && (o.home_spread || o.away_spread)) {
      return `${o.away_spread ?? ''} / ${o.home_spread ?? ''}`
    }
    return game?.odds_available ? 'Odds available' : 'Odds unavailable'
  })()

  const modelProbHome = analysis?.model_prob?.home
  const modelProbAway = analysis?.model_prob?.away
  const lean = analysis?.side?.lean
  const reasonsHome = analysis?.side?.reasons?.home || []
  const reasonsAway = analysis?.side?.reasons?.away || []
  const ev = analysis?.ev
  const odds = analysis?.ev?.odds
  const homeScore = analysis?.side?.home_score
  const awayScore = analysis?.side?.away_score
  const computedAt = analysis?.computed_at

  return (
    <div className="rounded-2xl border border-slate-700 bg-slate-900/70 p-4 flex flex-col gap-2 fade-in">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-sm text-slate-400">Game</div>
          <div className="text-lg font-semibold">
            {away} @ {home}
          </div>
        </div>
        <button
          onClick={analyze}
          className="rounded-xl bg-indigo-500/80 hover:bg-indigo-500 px-4 py-2 text-sm font-semibold transition"
          disabled={loading}
        >
          {loading ? 'Analyzing...' : 'Analyze'}
        </button>
      </div>
      <div className="text-sm text-slate-400">{oddsText}</div>
      {error && <div className="text-sm text-red-300">{error}</div>}
      {analysis && (
        <div className="text-sm text-slate-200 space-y-3">
          <div className="grid gap-2 md:grid-cols-2">
            <div className="rounded-xl border border-slate-700/60 bg-slate-800/60 p-3">
              <div className="flex flex-wrap gap-2 items-center mb-1">
                {lean && <span className="text-indigo-200 font-semibold">Lean: {lean}</span>}
                {computedAt && (
                  <span className="text-xs text-slate-400">Computed: {computedAt}</span>
                )}
              </div>
              <div className="flex flex-wrap gap-3">
                {homeScore !== undefined && <span>Home score: {homeScore.toFixed?.(2) ?? homeScore}</span>}
                {awayScore !== undefined && <span>Away score: {awayScore.toFixed?.(2) ?? awayScore}</span>}
                {modelProbHome !== undefined && (
                  <span>Home prob: {(modelProbHome * 100).toFixed(1)}%</span>
                )}
                {modelProbAway !== undefined && (
                  <span>Away prob: {(modelProbAway * 100).toFixed(1)}%</span>
                )}
              </div>
            </div>
            <div className="rounded-xl border border-slate-700/60 bg-slate-800/60 p-3">
              <div className="text-xs uppercase text-slate-400 mb-1">Odds / EV</div>
              {odds ? (
                <div className="space-y-1 text-slate-200">
                  <div>
                    Moneyline: Home {odds.home_ml ?? '--'} / Away {odds.away_ml ?? '--'}
                  </div>
                  {odds.home_spread !== undefined && odds.away_spread !== undefined && (
                    <div>
                      Spread: Home {odds.home_spread} / Away {odds.away_spread}
                    </div>
                  )}
                  {ev?.edge_pct && (
                    <div>
                      Edge: {(ev.edge_pct.home ?? 0).toFixed(2)}% / {(ev.edge_pct.away ?? 0).toFixed(2)}%
                    </div>
                  )}
                  {ev?.ev_units && (
                    <div>
                      EV units: {(ev.ev_units.home ?? 0).toFixed(3)} / {(ev.ev_units.away ?? 0).toFixed(3)}
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-slate-400">No odds/EV data</div>
              )}
            </div>
          </div>

          {(reasonsHome.length || reasonsAway.length) && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              <div>
                <div className="text-xs uppercase text-slate-400 mb-1">Home reasons</div>
                <ul className="list-disc list-inside text-slate-300 space-y-1">
                  {reasonsHome.map((r, i) => (
                    <li key={`h-${i}`}>{r}</li>
                  ))}
                </ul>
              </div>
              <div>
                <div className="text-xs uppercase text-slate-400 mb-1">Away reasons</div>
                <ul className="list-disc list-inside text-slate-300 space-y-1">
                  {reasonsAway.map((r, i) => (
                    <li key={`a-${i}`}>{r}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </div>
      )}

      {analysis && onAnalysis && (
        <button
          onClick={() => onAnalysis({ analysis, game })}
          className="self-start rounded-lg bg-slate-800 border border-slate-700 px-3 py-1 text-xs text-indigo-200 hover:border-indigo-400"
        >
          Open details
        </button>
      )}
    </div>
  )
}
