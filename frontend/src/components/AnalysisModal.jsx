import { createPortal } from 'react-dom'
import { useEffect } from 'react'
import { classNames } from '../utils/classNames'

function StatChip({ label, value }) {
  return (
    <div className="rounded-lg bg-slate-800/70 border border-slate-700 px-3 py-2 text-sm">
      <div className="text-xs uppercase text-slate-400">{label}</div>
      <div className="font-semibold text-slate-100">{value ?? 'N/A'}</div>
    </div>
  )
}

function listReasons(reasonsObj) {
  if (!reasonsObj) return []
  if (Array.isArray(reasonsObj)) return reasonsObj
  if (Array.isArray(reasonsObj.home)) return reasonsObj.home
  if (Array.isArray(reasonsObj.away)) return reasonsObj.away
  if (Array.isArray(reasonsObj.home_reasons)) return reasonsObj.home_reasons
  if (Array.isArray(reasonsObj.away_reasons)) return reasonsObj.away_reasons
  return []
}

export default function AnalysisModal({ open, onClose, game, analysis, sport }) {
  useEffect(() => {
    if (open) {
      const { scrollY } = window
      document.body.style.position = 'fixed'
      document.body.style.top = `-${scrollY}px`
      document.body.style.width = '100%'
      return () => {
        const top = document.body.style.top
        document.body.style.position = ''
        document.body.style.top = ''
        document.body.style.width = ''
        if (top) {
          const y = parseInt(top || '0', 10)
          window.scrollTo(0, -y)
        }
      }
    }
  }, [open])

  if (!open || !analysis) return null

  const home = game?.home_team || game?.home || 'HOME'
  const away = game?.away_team || game?.away || 'AWAY'
  const side = analysis?.side || {}
  const reasons = side?.reasons || {}
  const ev = analysis?.ev || {}
  const odds = ev?.odds || {}
  const edge = ev?.edge_pct || {}
  const evUnits = ev?.ev_units || {}
  const modelProb = analysis?.model_prob || {}
  const injuries = analysis?.injuries || {}
  const homeReasons = listReasons({ home: reasons?.home || reasons?.home_reasons })
  const awayReasons = listReasons({ away: reasons?.away || reasons?.away_reasons })

  const content = (
    <div className="fixed inset-0 z-50 flex items-start justify-center bg-black/60 p-4 overflow-auto">
      <div className="max-w-4xl w-full bg-slate-950 border border-slate-800 rounded-2xl shadow-2xl">
        <div className="flex items-center justify-between px-5 py-4 border-b border-slate-800">
          <div>
            <div className="text-sm text-slate-400">
              {away} @ {home}
            </div>
            <div className="text-lg font-semibold text-slate-100">
              Lean: {side?.lean || 'No clear edge'}
            </div>
            {analysis?.computed_at && (
              <div className="text-xs text-slate-500">Computed: {analysis.computed_at}</div>
            )}
          </div>
          <button
            onClick={onClose}
            className="rounded-lg bg-slate-800 hover:bg-slate-700 px-3 py-1 text-sm font-semibold"
          >
            Close
          </button>
        </div>

        <div className="grid gap-4 p-5 md:grid-cols-2">
          <div className="space-y-2">
            <div className="text-xs uppercase text-slate-400">Model</div>
            <div className="grid grid-cols-2 gap-2">
              <StatChip label="Home score" value={side?.home_score?.toFixed?.(2) ?? side?.home_score} />
              <StatChip label="Away score" value={side?.away_score?.toFixed?.(2) ?? side?.away_score} />
              <StatChip label="Home prob" value={modelProb?.home !== undefined ? `${(modelProb.home * 100).toFixed(1)}%` : 'N/A'} />
              <StatChip label="Away prob" value={modelProb?.away !== undefined ? `${(modelProb.away * 100).toFixed(1)}%` : 'N/A'} />
            </div>
          </div>

          <div className="space-y-2">
            <div className="text-xs uppercase text-slate-400">Odds / EV</div>
            <div className="grid grid-cols-2 gap-2">
              <StatChip label="Home ML" value={odds?.home_ml ?? '--'} />
              <StatChip label="Away ML" value={odds?.away_ml ?? '--'} />
              <StatChip label="Home edge" value={edge?.home !== undefined ? `${edge.home.toFixed(2)}%` : 'N/A'} />
              <StatChip label="Away edge" value={edge?.away !== undefined ? `${edge.away.toFixed(2)}%` : 'N/A'} />
              <StatChip label="Home EV" value={evUnits?.home !== undefined ? `${evUnits.home.toFixed(3)}u` : 'N/A'} />
              <StatChip label="Away EV" value={evUnits?.away !== undefined ? `${evUnits.away.toFixed(3)}u` : 'N/A'} />
            </div>
          </div>
        </div>

          <div className="grid gap-4 p-5 md:grid-cols-2">
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
              <div className="text-xs uppercase text-slate-400 mb-2">Home reasons</div>
              {homeReasons.length ? (
                <ul className="list-disc list-inside space-y-1 text-slate-200">
                {homeReasons.map((r, i) => (
                  <li key={`h-${i}`}>{r}</li>
                ))}
              </ul>
              ) : (
                <div className="text-sm text-slate-500">None</div>
            )}
          </div>
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
              <div className="text-xs uppercase text-slate-400 mb-2">Away reasons</div>
              {awayReasons.length ? (
                <ul className="list-disc list-inside space-y-1 text-slate-200">
                {awayReasons.map((r, i) => (
                  <li key={`a-${i}`}>{r}</li>
                ))}
              </ul>
              ) : (
                <div className="text-sm text-slate-500">None</div>
              )}
            </div>
          </div>

        {analysis?.goalies && (
          <div className="p-5 pt-0">
            <div className="text-xs uppercase text-slate-400 mb-2">Goalies</div>
            <div className="grid gap-3 md:grid-cols-2">
              <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-3">
                <div className="text-sm font-semibold text-slate-100">Home</div>
                <pre className="text-xs text-slate-400 whitespace-pre-wrap">
                  {JSON.stringify(analysis.goalies.home, null, 2)}
                </pre>
              </div>
              <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-3">
                <div className="text-sm font-semibold text-slate-100">Away</div>
                <pre className="text-xs text-slate-400 whitespace-pre-wrap">
                  {JSON.stringify(analysis.goalies.away, null, 2)}
                </pre>
              </div>
            </div>
          </div>
        )}

        {injuries && (injuries.home || injuries.away) && (
          <div className="p-5 pt-0">
            <div className="text-xs uppercase text-slate-400 mb-2">Injuries</div>
            <div className="grid gap-3 md:grid-cols-2">
              <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-3">
                <div className="text-sm font-semibold text-slate-100">Home</div>
                {Array.isArray(injuries.home) && injuries.home.length ? (
                  <ul className="list-disc list-inside text-xs text-slate-300 space-y-1">
                    {injuries.home.map((p, i) => (
                      <li key={`inj-h-${i}`}>{typeof p === 'string' ? p : `${p.player || p.name || 'Player'} - ${p.status || ''} ${p.notes || ''}`}</li>
                    ))}
                  </ul>
                ) : (
                  <div className="text-xs text-slate-500">None</div>
                )}
              </div>
              <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-3">
                <div className="text-sm font-semibold text-slate-100">Away</div>
                {Array.isArray(injuries.away) && injuries.away.length ? (
                  <ul className="list-disc list-inside text-xs text-slate-300 space-y-1">
                    {injuries.away.map((p, i) => (
                      <li key={`inj-a-${i}`}>{typeof p === 'string' ? p : `${p.player || p.name || 'Player'} - ${p.status || ''} ${p.notes || ''}`}</li>
                    ))}
                  </ul>
                ) : (
                  <div className="text-xs text-slate-500">None</div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )

  return createPortal(content, document.body)
}
