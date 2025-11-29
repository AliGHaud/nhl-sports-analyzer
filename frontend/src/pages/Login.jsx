import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

export default function Login() {
  const { loginWithGoogle, loginWithEmail, registerWithEmail, error, setError } = useAuth()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const navigate = useNavigate()

  const handleGoogle = async () => {
    try {
      await loginWithGoogle()
      navigate('/app')
    } catch (e) {
      setError?.(e?.message || 'Google login failed')
    }
  }

  const handleEmail = async () => {
    try {
      await loginWithEmail(email, password)
      navigate('/app')
    } catch (e) {
      setError?.(e?.message || 'Email login failed')
    }
  }

  const handleRegister = async () => {
    try {
      await registerWithEmail(email, password)
      navigate('/app')
    } catch (e) {
      setError?.(e?.message || 'Sign up failed')
    }
  }

  return (
    <div className="fade-in relative overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-br from-indigo-900/20 via-slate-950 to-purple-900/20 pointer-events-none" />
      <div className="absolute top-20 left-20 w-72 h-72 bg-indigo-600/20 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-20 right-20 w-96 h-96 bg-purple-600/20 rounded-full blur-3xl pointer-events-none" />

      <div className="relative min-h-screen flex items-center justify-center p-6 pt-20">
        <div className="relative z-10 w-full max-w-md">
          <div className="text-center mb-8">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center font-bold text-2xl mx-auto shadow-xl shadow-indigo-500/30 mb-4">
              SP
            </div>
            <h1 className="text-2xl font-bold">Welcome to Sports Predictor</h1>
            <p className="text-slate-400 mt-2">Sign in to access your picks</p>
          </div>

          <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-8 backdrop-blur">
            <button
              className="w-full py-3.5 rounded-xl bg-white text-slate-900 font-medium flex items-center justify-center gap-3 hover:bg-slate-100 transition mb-4"
              onClick={handleGoogle}
            >
              <svg className="w-5 h-5" viewBox="0 0 24 24"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/></svg>
              Continue with Google
            </button>

            <button
              className="w-full py-3.5 rounded-xl bg-slate-800 font-medium flex items-center justify-center gap-3 hover:bg-slate-700 transition mb-6"
              onClick={handleEmail}
            >
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>
              Continue with X
            </button>

            <div className="relative mb-6">
              <div className="absolute inset-0 flex items-center"><div className="w-full border-t border-slate-700"></div></div>
              <div className="relative flex justify-center text-sm"><span className="px-4 bg-slate-900 text-slate-400">or</span></div>
            </div>

            <div className="space-y-4">
              <div>
                <input
                  type="email"
                  className="w-full px-4 py-3 rounded-xl bg-slate-800/50 border border-slate-700 focus:border-indigo-500 outline-none transition"
                  placeholder="Email address"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                />
              </div>
              <div>
                <input
                  type="password"
                  className="w-full px-4 py-3 rounded-xl bg-slate-800/50 border border-slate-700 focus:border-indigo-500 outline-none transition"
                  placeholder="Password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                />
              </div>
              <button
                className="w-full py-3.5 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 font-medium hover:shadow-lg hover:shadow-indigo-500/25 transition"
                onClick={handleEmail}
              >
                Sign In
              </button>
              <button
                className="w-full py-3.5 rounded-xl border border-slate-700 bg-slate-900 text-slate-300 font-medium hover:border-indigo-400 hover:text-indigo-200 transition"
                onClick={handleRegister}
              >
                Sign Up Free
              </button>
            </div>

            <p className="text-center text-slate-400 text-sm mt-6">
              Don&apos;t have an account? <a href="#" className="text-indigo-400 hover:text-indigo-300">Sign up free</a>
            </p>
            {error && (
              <div className="mt-4 text-sm text-red-300 text-center">
                {error}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
