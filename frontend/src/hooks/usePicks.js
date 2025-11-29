import { useEffect, useState, useCallback } from 'react'
import { endpoints } from '../api/endpoints'
import { fetchWithAuth } from '../api/client'

const DEFAULT_STATS = {
  record: 'N/A',
  roi: 'N/A',
  streak: 'N/A',
  picksCount: 0,
}

export function usePicks(sport, betType) {
  const [picks, setPicks] = useState([])
  const [stats, setStats] = useState(DEFAULT_STATS)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState(null)

  const fetchPicks = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    try {
      const endpoint = endpoints?.[sport]?.[betType]
      if (!endpoint) {
        throw new Error('Endpoint not configured')
      }
      const res = await fetchWithAuth(endpoint)
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`)
      }
      const data = await res.json()
      const rawPicks = Array.isArray(data)
        ? data
        : data.picks || data.results || []
      const potd = data.potd ? [data.potd] : []
      const merged = potd.length
        ? [
            { ...potd[0], potd: true },
            ...rawPicks.filter(
              (p) =>
                p !== potd[0] &&
                !(p.home_team === potd[0].home_team && p.away_team === potd[0].away_team)
            ),
          ]
        : rawPicks

      setPicks(merged)
      setStats({
        record: data.record || DEFAULT_STATS.record,
        roi: data.roi || DEFAULT_STATS.roi,
        streak: data.streak || DEFAULT_STATS.streak,
        picksCount: merged.length,
      })
    } catch (err) {
      setError(err.message || 'Failed to load picks')
    } finally {
      setIsLoading(false)
    }
  }, [sport, betType])

  useEffect(() => {
    fetchPicks()
  }, [fetchPicks])

  return { picks, stats, isLoading, error, refetch: fetchPicks }
}
