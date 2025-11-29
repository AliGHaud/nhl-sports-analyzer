/**
 * Application-wide constants
 */

export const VIEW_MODES = {
  SIMPLE: 'simple',
  ADVANCED: 'advanced',
}

export const GRADE_COLORS = {
  A: 'emerald',
  'A-': 'emerald',
  'B+': 'green',
  B: 'lime',
  'B-': 'lime',
  'C+': 'yellow',
  C: 'yellow',
  'C-': 'orange',
  D: 'red',
  'No Bet': 'slate',
}

export const BREAKPOINTS = {
  SM: 640, // phone landscape, small tablets
  MD: 768, // tablets
  LG: 1024, // desktop
  XL: 1280, // large desktop
}

export const STORAGE_KEYS = {
  VIEW_MODE: 'sports_predictor_view_mode',
  DEBUG_MODE: 'sports_predictor_debug_mode',
}
