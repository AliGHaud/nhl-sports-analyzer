import { classNames } from '../utils/classNames'

function formatPct(val) {
  if (val === undefined || val === null) return 'N/A'
  const num = typeof val === 'number' ? val : parseFloat(val)
  if (Number.isNaN(num)) return 'N/A'
  return `${(num * (num < 1 ? 100 : 1)).toFixed(1)}%`
}

function formatEdge(edge, disagreement) {
  if (edge !== undefined && edge !== null) {
    const val = typeof edge === 'number' ? edge * (Math.abs(edge) < 1 ? 100 : 1) : parseFloat(edge)
    return Number.isNaN(val) ? 'N/A' : `${val.toFixed(2)}%`
  }
  if (disagreement !== undefined && disagreement !== null) {
    return `${disagreement > 0 ? '+' : ''}${disagreement.toFixed(1)} pts`
  }
  return 'N/A'
}

export default function POTDCard({ pick = {}, betType }) {
  const {
    home_team,
    away_team,
    pick_side,
    spread,
    edge,
    ev,
    model_prob,
    elo_spread,
    market_spread,
    disagreement,
  } = pick

  const matchup = `${away_team || 'AWY'} @ ${home_team || 'HOME'}`
  const pickText =
    pick_side === 'home'
      ? `${home_team || 'HOME'} ${spread ? `${spread > 0 ? '+' : ''}${spread}` : ''}`
      : `${away_team || 'AWY'} ${spread ? `${spread > 0 ? '+' : ''}${spread}` : ''}`

  return (
    <div className="rounded-2xl border border-amber-300/40 bg-gradient-to-br from-amber-400/20 via-orange-400/15 to-amber-500/10 p-5 shadow-lg shadow-amber-500/20 mb-4 fade-in">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2 text-amber-200 font-semibold">
          <span className="text-xl">⭐</span>
          Pick of the Day
        </div>
        <span className="text-xs font-bold text-amber-100 bg-amber-500/30 px-3 py-1 rounded-full">
          BEST BET
        </span>
      </div>
      <div className="text-sm text-amber-100/80 mb-1">{matchup}</div>
      <div className="text-2xl font-bold mb-3">{pickText}</div>
      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
        <InfoChip label="Model" value={formatPct(model_prob)} />
        <InfoChip label="Edge" value={formatEdge(edge, disagreement)} />
        <InfoChip label="EV" value={ev !== undefined ? `${ev.toFixed(3)}u` : 'N/A'} />
        {betType === 'spreads' && (
          <>
            <InfoChip
              label="Elo Spread"
              value={elo_spread !== undefined ? elo_spread.toFixed(1) : 'N/A'}
            />
            <InfoChip
              label="Market"
              value={market_spread !== undefined ? market_spread.toFixed(1) : 'N/A'}
            />
          </>
        )}
      </div>
    </div>
  )
}

function InfoChip({ label, value }) {
  return (
    <div
      className={classNames(
        'rounded-xl border border-white/20 bg-white/5',
        'px-3 py-2 text-sm flex flex-col gap-1'
      )}
    >
      <span className="text-xs uppercase tracking-wide text-white/70">{label}</span>
      <span className="font-semibold">{value}</span>
    </div>
  )
}
