import { initializeApp } from 'firebase/app'
import { getAuth } from 'firebase/auth'

const firebaseConfig = {
  apiKey: 'AIzaSyAd3w0--yeSJm4pUAbh2Oy8qpIf6OD8fc4',
  authDomain: 'sports-analyzer-bc0d6.firebaseapp.com',
  projectId: 'sports-analyzer-bc0d6',
  storageBucket: 'sports-analyzer-bc0d6.firebasestorage.app',
  messagingSenderId: '773945640566',
  appId: '1:773945640566:web:c4c0a9ccf1e30df7e4c157',
}

const app = initializeApp(firebaseConfig)
export const auth = getAuth(app)
