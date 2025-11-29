import { createContext, useContext } from 'react'
import { useLocalStorage } from '../hooks/useLocalStorage'
import { VIEW_MODES, STORAGE_KEYS } from '../utils/constants'

const ViewModeContext = createContext()

/**
 * Hook to access view mode context
 * @returns {{ viewMode: string, setViewMode: function, isAdvanced: boolean, isSimple: boolean }}
 */
export function useViewMode() {
  const context = useContext(ViewModeContext)
  if (!context) {
    throw new Error('useViewMode must be used within ViewModeProvider')
  }
  return context
}

/**
 * Provider component for view mode state
 * Manages simple vs advanced view mode with localStorage persistence
 */
export function ViewModeProvider({ children }) {
  const [viewMode, setViewMode] = useLocalStorage(
    STORAGE_KEYS.VIEW_MODE,
    VIEW_MODES.SIMPLE
  )

  const value = {
    viewMode,
    setViewMode,
    isAdvanced: viewMode === VIEW_MODES.ADVANCED,
    isSimple: viewMode === VIEW_MODES.SIMPLE,
  }

  return <ViewModeContext.Provider value={value}>{children}</ViewModeContext.Provider>
}
