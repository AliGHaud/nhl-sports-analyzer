import { Routes, Route, Link, useLocation } from 'react-router-dom'
import Picks from './pages/Picks'
import Home from './pages/Home'
import Login from './pages/Login'
import BetTracker from './pages/BetTracker'
import Alerts from './pages/Alerts'
import Upgrade from './pages/Upgrade'
import { ViewModeProvider } from './context/ViewModeContext'
import { AuthProvider } from './context/AuthContext'
import ProtectedRoute from './components/ProtectedRoute'
import { useAuth } from './context/AuthContext'

function NavBar() {
  const location = useLocation()
  const { user, userTier, logout } = useAuth()
  const isPro = userTier === 'pro'
  const links = [
    { to: '/', label: 'Home', private: false },
    { to: '/login', label: 'Login', private: false },
    { to: '/app', label: 'Picks', private: true },
    { to: '/tracker', label: 'Bet Tracker', private: true },
    { to: '/alerts', label: 'Alerts', private: true },
  ]
  return (
    <div className="sticky top-0 z-40 bg-slate-950/80 backdrop-blur border-b border-slate-800">
      <div className="max-w-6xl mx-auto px-4 py-3 flex items-center gap-3 text-sm">
        <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 flex items-center justify-center font-black text-base shadow-lg shadow-indigo-500/40">
          SP
        </div>
        <div className="text-lg font-semibold text-slate-100">Sports Predictor</div>
        <div className="flex gap-2 ml-auto items-center">
          {links
            .filter((link) => {
              if (link.private && !user) return false
              if (!link.private && user && link.to === '/login') return false
              return true
            })
            .map((link) => {
              const active = location.pathname === link.to
              return (
                <Link
                  key={link.to}
                  to={link.to}
                  className={[
                    'px-3 py-1 rounded-lg border text-xs font-semibold transition',
                    active
                      ? 'bg-indigo-500/20 border-indigo-400 text-white'
                      : 'bg-slate-900 border-slate-700 text-slate-300 hover:border-indigo-400 hover:text-indigo-200',
                  ].join(' ')}
                >
                  {link.label}
                </Link>
              )
            })}
          {user ? (
            <>
              {!isPro && (
                <Link
                  to="/upgrade"
                  className="rounded-lg bg-gradient-to-r from-indigo-500 to-purple-600 text-white px-3 py-1 text-xs font-semibold shadow hover:from-indigo-400 hover:to-purple-500 transition"
                >
                  Upgrade
                </Link>
              )}
              <span className="text-xs text-slate-300 px-2">{user.email || 'Logged in'}</span>
              <button
                onClick={logout}
                className="rounded-lg bg-slate-800 border border-slate-700 px-3 py-1 text-xs font-semibold text-slate-200 hover:border-indigo-400 hover:text-indigo-200 transition"
              >
                Logout
              </button>
            </>
          ) : (
            <Link
              to="/login"
              className="rounded-lg bg-indigo-500/80 text-white px-3 py-1 text-xs font-semibold border border-indigo-400 hover:bg-indigo-500 transition"
            >
              Get Started
            </Link>
          )}
        </div>
      </div>
    </div>
  )
}

function App() {
  return (
    <AuthProvider>
      <ViewModeProvider>
        <div className="min-h-screen bg-slate-950 text-slate-100">
          <NavBar />
          <div className="max-w-6xl mx-auto px-4 py-6">
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/login" element={<Login />} />
              <Route path="/upgrade" element={<Upgrade />} />
              <Route
                path="/app"
                element={
                  <ProtectedRoute>
                    <Picks />
                  </ProtectedRoute>
                }
              />
              <Route
                path="/tracker"
                element={
                  <ProtectedRoute>
                    <BetTracker />
                  </ProtectedRoute>
                }
              />
              <Route
                path="/alerts"
                element={
                  <ProtectedRoute>
                    <Alerts />
                  </ProtectedRoute>
                }
              />
              <Route path="*" element={<Home />} />
            </Routes>
          </div>
        </div>
      </ViewModeProvider>
    </AuthProvider>
  )
}

export default App
