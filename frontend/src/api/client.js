import { auth } from '../firebase'

export async function fetchWithAuth(url, options = {}, explicitToken) {
  const token = explicitToken || (await auth.currentUser?.getIdToken?.())
  const headers = {
    ...(options.headers || {}),
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  }
  return fetch(url, { ...options, headers })
}
