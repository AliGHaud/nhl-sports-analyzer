import { useState, useEffect } from 'react'

/**
 * Custom hook to sync state with localStorage
 * @param {string} key - localStorage key
 * @param {*} initialValue - default value if key doesn't exist
 * @returns {[*, function]} - [storedValue, setValue]
 */
export function useLocalStorage(key, initialValue) {
  // Get initial value from localStorage or use provided default
  const [storedValue, setStoredValue] = useState(() => {
    try {
      const item = window.localStorage.getItem(key)
      return item ? JSON.parse(item) : initialValue
    } catch (error) {
      console.warn(`Error reading localStorage key "${key}":`, error)
      return initialValue
    }
  })

  // Update localStorage when state changes
  useEffect(() => {
    try {
      window.localStorage.setItem(key, JSON.stringify(storedValue))
    } catch (error) {
      console.warn(`Error setting localStorage key "${key}":`, error)
    }
  }, [key, storedValue])

  return [storedValue, setStoredValue]
}
