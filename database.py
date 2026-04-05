"""
database.py — MongoDB connection manager.
Handles connecting to MongoDB Atlas or local MongoDB instance.
"""

import os
import streamlit as st
from datetime import datetime

try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    HAS_PYMONGO = True
except ImportError:
    HAS_PYMONGO = False


# ── MongoDB URI — set via environment variable or .env file ──
# Local:  mongodb://localhost:27017
# Atlas:  mongodb+srv://<user>:<pass>@cluster.mongodb.net/automl_studio
MONGO_URI = os.environ.get(
    "mongodb+srv://RouteX:@s23D21A6650@cluster0.8nmihjs.mongodb.net/?appName=Cluster0",
    "mongodb://localhost:27017"          # default: local MongoDB
)
DB_NAME = os.environ.get("MONGO_DB_NAME", "automl_studio")


class MongoDB:
    """Singleton MongoDB connection wrapper."""

    _client = None
    _db     = None

    @classmethod
    def get_db(cls):
        if not HAS_PYMONGO:
            st.error("pymongo is not installed. Run: pip install pymongo")
            return None

        if cls._client is None:
            try:
                cls._client = MongoClient(
                    MONGO_URI,
                    serverSelectionTimeoutMS=5000,   # 5-second timeout
                    connectTimeoutMS=5000,
                )
                # Ping to verify connection
                cls._client.admin.command("ping")
                cls._db = cls._client[DB_NAME]

                # ── Create indexes on first connect ──────────────
                cls._db["users"].create_index("username", unique=True)
                cls._db["users"].create_index("email",    unique=True)
                cls._db["sessions"].create_index("username")
                cls._db["sessions"].create_index("created_at")
                cls._db["history"].create_index([("username", 1), ("created_at", -1)])

            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                st.error(
                    f"❌ Cannot connect to MongoDB at `{MONGO_URI}`.\n\n"
                    f"**Steps to fix:**\n"
                    f"1. Make sure MongoDB is running (`mongod`)\n"
                    f"2. Or set your Atlas URI: `MONGO_URI=mongodb+srv://...`\n\n"
                    f"Error: {e}"
                )
                return None

        return cls._db

    @classmethod
    def users(cls):
        db = cls.get_db()
        return db["users"] if db is not None else None

    @classmethod
    def sessions(cls):
        db = cls.get_db()
        return db["sessions"] if db is not None else None

    @classmethod
    def history(cls):
        db = cls.get_db()
        return db["history"] if db is not None else None

    @classmethod
    def close(cls):
        if cls._client:
            cls._client.close()
            cls._client = None
            cls._db     = None
