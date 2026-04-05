"""
auth_manager.py — User authentication logic.
Handles registration, login, password hashing, session tokens.
"""

import os
import re
import hashlib
import secrets
import string
from datetime import datetime, timedelta
from typing import Optional

try:
    import bcrypt
    HAS_BCRYPT = True
except ImportError:
    HAS_BCRYPT = False

from backend.auth.database import MongoDB


# ── Password policy ───────────────────────────────────────────
MIN_PASSWORD_LEN = 8
USERNAME_REGEX   = re.compile(r'^[a-zA-Z0-9_]{3,30}$')
EMAIL_REGEX      = re.compile(r'^[\w\.-]+@[\w\.-]+\.\w{2,}$')

# ── Session TTL ───────────────────────────────────────────────
SESSION_TTL_HOURS = 24 * 7     # 7 days


class AuthManager:
    """Handles all authentication operations."""

    # ─────────────────────────────────────────────────────────
    # PASSWORD HASHING
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt (preferred) or SHA-256 fallback."""
        if HAS_BCRYPT:
            salt = bcrypt.gensalt(rounds=12)
            return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")
        else:
            # Fallback: salted SHA-256
            salt = secrets.token_hex(32)
            hashed = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
            return f"sha256:{salt}:{hashed}"

    @staticmethod
    def verify_password(password: str, stored_hash: str) -> bool:
        """Verify password against stored hash."""
        if HAS_BCRYPT and not stored_hash.startswith("sha256:"):
            try:
                return bcrypt.checkpw(
                    password.encode("utf-8"),
                    stored_hash.encode("utf-8")
                )
            except Exception:
                return False
        else:
            # SHA-256 fallback
            parts = stored_hash.split(":")
            if len(parts) != 3:
                return False
            _, salt, original_hash = parts
            check_hash = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
            return secrets.compare_digest(check_hash, original_hash)

    # ─────────────────────────────────────────────────────────
    # VALIDATION
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def validate_username(username: str) -> tuple[bool, str]:
        if not USERNAME_REGEX.match(username):
            return False, "Username must be 3–30 chars, letters/numbers/underscore only."
        return True, ""

    @staticmethod
    def validate_email(email: str) -> tuple[bool, str]:
        if not EMAIL_REGEX.match(email):
            return False, "Invalid email address format."
        return True, ""

    @staticmethod
    def validate_password(password: str) -> tuple[bool, str]:
        if len(password) < MIN_PASSWORD_LEN:
            return False, f"Password must be at least {MIN_PASSWORD_LEN} characters."
        if not any(c.isupper() for c in password):
            return False, "Password must contain at least one uppercase letter."
        if not any(c.isdigit() for c in password):
            return False, "Password must contain at least one number."
        return True, ""

    # ─────────────────────────────────────────────────────────
    # REGISTRATION
    # ─────────────────────────────────────────────────────────

    @classmethod
    def register(cls, username: str, email: str, password: str,
                 full_name: str = "") -> tuple[bool, str]:
        """Register a new user. Returns (success, message)."""

        # Validate inputs
        ok, msg = cls.validate_username(username)
        if not ok: return False, msg

        ok, msg = cls.validate_email(email)
        if not ok: return False, msg

        ok, msg = cls.validate_password(password)
        if not ok: return False, msg

        users = MongoDB.users()
        if users is None:
            return False, "Database connection failed."

        # Check uniqueness
        if users.find_one({"username": username.lower()}):
            return False, "Username already exists. Please choose another."
        if users.find_one({"email": email.lower()}):
            return False, "An account with this email already exists."

        # Create user document
        user_doc = {
            "username":   username.lower(),
            "email":      email.lower(),
            "full_name":  full_name.strip(),
            "password_hash": cls.hash_password(password),
            "role":       "user",            # user | admin
            "is_active":  True,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "last_login": None,
            "login_count": 0,
            "profile": {
                "avatar_color": cls._random_avatar_color(),
                "bio": "",
                "organization": "",
            },
            "stats": {
                "total_runs":     0,
                "total_models":   0,
                "datasets_uploaded": 0,
            }
        }

        users.insert_one(user_doc)
        return True, "Account created successfully! You can now log in."

    # ─────────────────────────────────────────────────────────
    # LOGIN
    # ─────────────────────────────────────────────────────────

    @classmethod
    def login(cls, username_or_email: str, password: str) -> tuple[bool, str, Optional[dict]]:
        """Authenticate user. Returns (success, message, user_doc)."""

        users = MongoDB.users()
        if users is None:
            return False, "Database connection failed.", None

        val = username_or_email.lower().strip()

        # Find by username or email
        user = users.find_one({"username": val}) or \
               users.find_one({"email": val})

        if not user:
            return False, "No account found with that username or email.", None

        if not user.get("is_active", True):
            return False, "Your account has been deactivated. Contact support.", None

        if not cls.verify_password(password, user["password_hash"]):
            return False, "Incorrect password. Please try again.", None

        # Update login stats
        users.update_one(
            {"_id": user["_id"]},
            {"$set": {"last_login": datetime.utcnow(),
                       "updated_at": datetime.utcnow()},
             "$inc": {"login_count": 1}}
        )

        # Return sanitized user (no password hash)
        safe_user = {k: v for k, v in user.items() if k != "password_hash"}
        safe_user["_id"] = str(safe_user["_id"])
        return True, "Login successful!", safe_user

    # ─────────────────────────────────────────────────────────
    # SESSION TOKENS
    # ─────────────────────────────────────────────────────────

    @classmethod
    def create_session(cls, username: str) -> str:
        """Create a session token and store in DB. Returns token."""
        token = secrets.token_urlsafe(48)
        sessions = MongoDB.sessions()
        if sessions is None:
            return token   # still return token even if DB fails

        sessions.insert_one({
            "token":      token,
            "username":   username,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(hours=SESSION_TTL_HOURS),
            "is_active":  True,
        })
        return token

    @classmethod
    def validate_session(cls, token: str) -> Optional[dict]:
        """Validate session token. Returns user doc or None."""
        if not token:
            return None

        sessions = MongoDB.sessions()
        if sessions is None:
            return None

        session = sessions.find_one({
            "token":     token,
            "is_active": True,
            "expires_at": {"$gt": datetime.utcnow()},
        })
        if not session:
            return None

        # Get user
        users = MongoDB.users()
        user  = users.find_one({"username": session["username"]})
        if not user:
            return None

        safe = {k: v for k, v in user.items() if k != "password_hash"}
        safe["_id"] = str(safe["_id"])
        return safe

    @classmethod
    def logout(cls, token: str):
        """Invalidate a session token."""
        sessions = MongoDB.sessions()
        if sessions:
            sessions.update_one(
                {"token": token},
                {"$set": {"is_active": False}}
            )

    # ─────────────────────────────────────────────────────────
    # PASSWORD RESET
    # ─────────────────────────────────────────────────────────

    @classmethod
    def change_password(cls, username: str, old_password: str,
                        new_password: str) -> tuple[bool, str]:
        """Change user password after verifying old password."""
        users = MongoDB.users()
        if users is None:
            return False, "Database connection failed."

        user = users.find_one({"username": username.lower()})
        if not user:
            return False, "User not found."

        if not cls.verify_password(old_password, user["password_hash"]):
            return False, "Current password is incorrect."

        ok, msg = cls.validate_password(new_password)
        if not ok:
            return False, msg

        users.update_one(
            {"username": username.lower()},
            {"$set": {
                "password_hash": cls.hash_password(new_password),
                "updated_at":    datetime.utcnow(),
            }}
        )
        return True, "Password changed successfully."

    # ─────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _random_avatar_color() -> str:
        colors = ["#6C63FF","#43E97B","#FF6584","#FFD700","#FF8C42","#A8DADC","#E63946"]
        import random
        return random.choice(colors)

    @classmethod
    def get_user(cls, username: str) -> Optional[dict]:
        users = MongoDB.users()
        if users is None:
            return None
        user = users.find_one({"username": username.lower()})
        if not user:
            return None
        safe = {k: v for k, v in user.items() if k != "password_hash"}
        safe["_id"] = str(safe["_id"])
        return safe

    @classmethod
    def update_profile(cls, username: str, updates: dict) -> tuple[bool, str]:
        """Update user profile fields."""
        users = MongoDB.users()
        if users is None:
            return False, "Database error."
        allowed = {"full_name", "profile.bio", "profile.organization", "profile.avatar_color"}
        safe_updates = {k: v for k, v in updates.items() if k in allowed}
        safe_updates["updated_at"] = datetime.utcnow()
        users.update_one({"username": username}, {"$set": safe_updates})
        return True, "Profile updated."

    @classmethod
    def update_stats(cls, username: str, stat: str, increment: int = 1):
        """Increment a user stat (total_runs, total_models, etc.)."""
        users = MongoDB.users()
        if users:
            users.update_one(
                {"username": username},
                {"$inc": {f"stats.{stat}": increment}}
            )
