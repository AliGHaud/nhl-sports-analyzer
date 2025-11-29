import { useCallback, useEffect, useState } from 'react'
import { endpoints } from '../api/endpoints'
import { fetchWithAuth } from '../api/client'

export function useSchedule(sport) {
  const [games, setGames] = useState([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState(null)

  const fetchSchedule = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    try {
      const endpoint = endpoints?.[sport]?.schedule
      if (!endpoint) throw new Error('Schedule endpoint not configured')
      const res = await fetchWithAuth(endpoint)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      // data may be list or object with "games"
      const list = Array.isArray(data) ? data : data.games || []
      setGames(list)
    } catch (err) {
      setError(err.message || 'Failed to load schedule')
    } finally {
      setIsLoading(false)
    }
  }, [sport])

  useEffect(() => {
    fetchSchedule()
  }, [fetchSchedule])

  return { games, isLoading, error, refetch: fetchSchedule }
}
