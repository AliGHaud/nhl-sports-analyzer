import { useState, useMemo, useCallback } from 'react'
import Header from '../components/Header'
import SportTabs from '../components/SportTabs'
import BetTypeTabs from '../components/BetTypeTabs'
import StatsBar from '../components/StatsBar'
import POTDCard from '../components/POTDCard'
import PickCard from '../components/PickCard'
import LoadingSkeleton from '../components/LoadingSkeleton'
import Toast from '../components/Toast'
import ViewModeToggle from '../components/ViewModeToggle'
import { usePicks } from '../hooks/usePicks'
import { useSchedule } from '../hooks/useSchedule'
import TodayList from '../components/TodayList'
import AnalysisModal from '../components/AnalysisModal'
import ProFeature from '../components/ProFeature'
import { useAuth } from '../context/AuthContext'
import ProBadge from '../components/ProBadge'

export default function Picks() {
  const [sport, setSport] = useState('nhl')
  const [betType, setBetType] = useState('moneyline')
  const [toast, setToast] = useState('')
  const [selectedAnalysis, setSelectedAnalysis] = useState(null)
  const { userTier } = useAuth()
  const { picks, stats, isLoading, error, refetch } = usePicks(sport, betType)
   const { games, isLoading: loadingGames, error: gamesError, refetch: refetchGames } = useSchedule(sport)

  const { potd, others } = useMemo(() => {
    const pickList = picks || []
    const top = pickList.find((p) => p.potd) || pickList[0]
    const rest = top ? pickList.filter((p) => p !== top) : []
    return { potd: top, others: rest }
  }, [picks])

  const handleRefresh = async () => {
    await refetch()
    await refetchGames()
    setToast('Picks refreshed')
    setTimeout(() => setToast(''), 2000)
  }

  const handleAnalysis = useCallback(
    (payload) => {
      setSelectedAnalysis((prev) => {
        if (prev && prev.game && payload?.game) {
          const prevKey = `${prev.game.home_team || prev.game.home}-${prev.game.away_team || prev.game.away}`
          const newKey = `${payload.game.home_team || payload.game.home}-${payload.game.away_team || payload.game.away}`
          if (prevKey === newKey) {
            return null
          }
        }
        return payload
      })
    },
    []
  )

  return (
    <div className="fade-in">
      <Header onRefresh={handleRefresh} />
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-4">
          <SportTabs active={sport} onChange={setSport} />
          <BetTypeTabs active={betType} onChange={setBetType} />
        </div>
        <ViewModeToggle />
      </div>
      {userTier !== 'pro' && (
        <div className="mb-3 rounded-xl border border-indigo-400/50 bg-indigo-500/10 px-4 py-3 text-sm text-indigo-100">
          <div className="flex items-center gap-2">
            <div className="font-semibold text-indigo-100">Pro feature</div>
            <ProBadge />
          </div>
          <div className="text-indigo-200/90">
            Picks and spreads are locked for free accounts. Upgrade to Pro to unlock full access.
          </div>
          <div className="mt-2">
            <a
              href="/upgrade"
              className="inline-flex items-center rounded-lg bg-gradient-to-r from-indigo-500 to-purple-600 px-3 py-1 text-xs font-semibold text-white shadow hover:from-indigo-400 hover:to-purple-500 transition"
            >
              Upgrade to Pro
            </a>
          </div>
        </div>
      )}
      <StatsBar {...stats} />

      {error && (
        <div className="mb-3 rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-200">
          {error}
        </div>
      )}

          {isLoading ? (
            <LoadingSkeleton />
          ) : (
            <ProFeature userTier={userTier} blur>
              <>
                {potd && <POTDCard pick={potd} betType={betType} />}
                <div className="grid gap-3">
                  {others.map((p, idx) => (
                    <PickCard key={`${p.home_team}-${p.away_team}-${idx}`} pick={p} rank={idx + 2} betType={betType} />
                  ))}
                </div>
                {!potd && <div className="text-slate-400 text-sm mt-4">No picks found.</div>}
              </>
            </ProFeature>
          )}

      <TodayList
        sport={sport}
        games={games}
        isLoading={loadingGames}
        error={gamesError}
        onRefresh={refetchGames}
        onAnalysis={handleAnalysis}
      />

      <Toast message={toast} />
      <AnalysisModal
        open={!!selectedAnalysis}
        onClose={() => setSelectedAnalysis(null)}
        game={selectedAnalysis?.game}
        analysis={selectedAnalysis?.analysis}
        sport={sport}
      />
    </div>
  )
}
