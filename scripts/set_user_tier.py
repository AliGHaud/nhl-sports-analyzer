"""
Grant or revoke Pro tier for a user.

Usage:
  python scripts/set_user_tier.py <email> pro
  python scripts/set_user_tier.py <email> free
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import firebase_admin
from firebase_admin import auth, credentials

FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase-credentials.json")


def init_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
        firebase_admin.initialize_app(cred)


def set_user_tier(email: str, tier: str):
    init_firebase()

    try:
        user = auth.get_user_by_email(email)
    except auth.UserNotFoundError:
        print(f"Error: No user found with email {email}")
        return False

    auth.set_custom_user_claims(user.uid, {"tier": tier})
    print(f"Success: {email} is now '{tier}'")
    print(f"UID: {user.uid}")
    print("Note: User must log out and back in for changes to take effect.")
    return True


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/set_user_tier.py <email> <tier>")
        print("  tier: 'pro' or 'free'")
        sys.exit(1)

    email = sys.argv[1]
    tier = sys.argv[2].lower()

    if tier not in ("pro", "free"):
        print("Error: tier must be 'pro' or 'free'")
        sys.exit(1)

    set_user_tier(email, tier)
