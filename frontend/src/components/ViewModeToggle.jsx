import { useViewMode } from '../context/ViewModeContext'
import { VIEW_MODES } from '../utils/constants'
import { classNames } from '../utils/classNames'

/**
 * Toggle button to switch between Simple and Advanced view modes
 * Displays as a segmented control with two options
 */
export default function ViewModeToggle() {
  const { viewMode, setViewMode } = useViewMode()

  return (
    <div className="inline-flex items-center gap-1 p-1 rounded-xl bg-slate-800/70 border border-slate-700">
      <button
        onClick={() => setViewMode(VIEW_MODES.SIMPLE)}
        className={classNames(
          'px-4 py-2 rounded-lg text-sm font-medium transition-all',
          viewMode === VIEW_MODES.SIMPLE
            ? 'bg-indigo-600 text-white shadow-md'
            : 'text-slate-400 hover:text-slate-300'
        )}
        aria-pressed={viewMode === VIEW_MODES.SIMPLE}
      >
        Simple
      </button>
      <button
        onClick={() => setViewMode(VIEW_MODES.ADVANCED)}
        className={classNames(
          'px-4 py-2 rounded-lg text-sm font-medium transition-all',
          viewMode === VIEW_MODES.ADVANCED
            ? 'bg-indigo-600 text-white shadow-md'
            : 'text-slate-400 hover:text-slate-300'
        )}
        aria-pressed={viewMode === VIEW_MODES.ADVANCED}
      >
        Advanced
      </button>
    </div>
  )
}
