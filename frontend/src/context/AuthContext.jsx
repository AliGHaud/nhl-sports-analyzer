import { createContext, useContext, useEffect, useState, useCallback } from 'react'
import {
  GoogleAuthProvider,
  onAuthStateChanged,
  signInWithPopup,
  signOut,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
} from 'firebase/auth'
import { auth } from '../firebase'
import { fetchWithAuth } from '../api/client'

const AuthContext = createContext(null)

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) {
    throw new Error('useAuth must be used within AuthProvider')
  }
  return ctx
}

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [idToken, setIdToken] = useState(null)
  const [userTier, setUserTier] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const refreshTier = useCallback(async (tokenOverride) => {
    setUserTier(null)
    try {
      const res = await fetchWithAuth('/auth/me', {}, tokenOverride)
      if (res.ok) {
        const data = await res.json()
        setUserTier(data?.tier || null)
      }
    } catch (_) {
      /* ignore */
    }
  }, [])

  useEffect(() => {
    const unsub = onAuthStateChanged(auth, async (firebaseUser) => {
      setUser(firebaseUser)
      if (firebaseUser) {
        try {
          const token = await firebaseUser.getIdToken()
          setIdToken(token)
          await refreshTier(token)
        } catch (e) {
          setError(e?.message || 'Failed to get token')
        }
      } else {
        setIdToken(null)
        setUserTier(null)
      }
      setLoading(false)
    })
    return () => unsub()
  }, [refreshTier])

  const loginWithGoogle = async () => {
    setError(null)
    const provider = new GoogleAuthProvider()
    await signInWithPopup(auth, provider)
  }

  const loginWithEmail = async (email, password) => {
    setError(null)
    await signInWithEmailAndPassword(auth, email, password)
  }

  const registerWithEmail = async (email, password) => {
    setError(null)
    await createUserWithEmailAndPassword(auth, email, password)
  }

  const logout = async () => {
    setError(null)
    await signOut(auth)
  }

  const value = {
    user,
    idToken,
    userTier,
    loading,
    error,
    loginWithGoogle,
    loginWithEmail,
    registerWithEmail,
    logout,
    setError,
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}
