"""
history_manager.py — Save and retrieve per-user pipeline run history.
"""

from datetime import datetime
from typing import Optional
from backend.auth.database import MongoDB


class HistoryManager:
    """Stores pipeline run history per user in MongoDB."""

    @classmethod
    def save_run(cls, username: str, run_data: dict) -> Optional[str]:
        """Save a pipeline run record. Returns inserted ID string."""
        history = MongoDB.history()
        if history is None:
            return None

        doc = {
            "username":     username,
            "created_at":   datetime.utcnow(),
            "dataset_name": run_data.get("dataset_name", "Unknown"),
            "n_rows":       run_data.get("n_rows", 0),
            "n_cols":       run_data.get("n_cols", 0),
            "task_type":    run_data.get("task_type", "Unknown"),
            "target_col":   run_data.get("target_col", None),
            "best_model":   run_data.get("best_model", None),
            "metrics":      run_data.get("metrics", {}),
            "models_trained": run_data.get("models_trained", []),
            "feature_names":  run_data.get("feature_names", []),
            "leaderboard":    run_data.get("leaderboard", []),
            "notes":          run_data.get("notes", ""),
            "status":         "completed",
        }

        result = history.insert_one(doc)
        # Update user stats
        try:
            from backend.auth.auth_manager import AuthManager
            AuthManager.update_stats(username, "total_runs")
            n_models = len(run_data.get("models_trained", []))
            if n_models:
                AuthManager.update_stats(username, "total_models", n_models)
            AuthManager.update_stats(username, "datasets_uploaded")
        except Exception:
            pass

        return str(result.inserted_id)

    @classmethod
    def get_user_history(cls, username: str, limit: int = 50) -> list:
        """Fetch all pipeline runs for a user, newest first."""
        history = MongoDB.history()
        if history is None:
            return []

        cursor = history.find(
            {"username": username},
            sort=[("created_at", -1)],
            limit=limit
        )
        runs = []
        for doc in cursor:
            doc["_id"] = str(doc["_id"])
            runs.append(doc)
        return runs

    @classmethod
    def get_run(cls, run_id: str) -> Optional[dict]:
        """Fetch a single run by its ID."""
        from bson import ObjectId
        history = MongoDB.history()
        if history is None:
            return None
        try:
            doc = history.find_one({"_id": ObjectId(run_id)})
            if doc:
                doc["_id"] = str(doc["_id"])
            return doc
        except Exception:
            return None

    @classmethod
    def delete_run(cls, run_id: str, username: str) -> bool:
        """Delete a run (only if owned by username)."""
        from bson import ObjectId
        history = MongoDB.history()
        if history is None:
            return False
        try:
            result = history.delete_one({
                "_id": ObjectId(run_id),
                "username": username
            })
            return result.deleted_count > 0
        except Exception:
            return False

    @classmethod
    def get_stats(cls, username: str) -> dict:
        """Aggregate stats for a user's history."""
        history = MongoDB.history()
        if history is None:
            return {}

        runs = list(history.find({"username": username}))
        if not runs:
            return {"total_runs": 0}

        task_counts = {}
        best_models = {}
        for r in runs:
            t = r.get("task_type", "Unknown")
            task_counts[t] = task_counts.get(t, 0) + 1
            bm = r.get("best_model")
            if bm:
                best_models[bm] = best_models.get(bm, 0) + 1

        favourite_model = max(best_models, key=best_models.get) if best_models else "N/A"

        return {
            "total_runs":     len(runs),
            "task_breakdown": task_counts,
            "favourite_model": favourite_model,
            "first_run":      min(r["created_at"] for r in runs).strftime("%Y-%m-%d"),
            "last_run":       max(r["created_at"] for r in runs).strftime("%Y-%m-%d"),
        }
