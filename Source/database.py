"""
Database layer.
Handles all SQLite operations and provides a clean API to the rest of the app.
"""

import sqlite3
import numpy as np
import os
from datetime import datetime
from typing import Optional
from config import DB_PATH, DATA_DIR


def _adapt_array(arr: np.ndarray) -> bytes:
    """Convert numpy array to bytes for SQLite storage."""
    return arr.astype(np.float32).tobytes()


def _convert_array(blob: bytes) -> np.ndarray:
    """Convert bytes from SQLite back to numpy array."""
    return np.frombuffer(blob, dtype=np.float32)


class Database:
    """SQLite database manager for face recognition data."""

    def __init__(self, db_path: str = DB_PATH):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Create tables if they don't exist."""
        schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
        with open(schema_path, "r") as f:
            schema = f.read()
        with self._get_conn() as conn:
            conn.executescript(schema)

    # ------------------------------------------------------------------
    # Person CRUD
    # ------------------------------------------------------------------

    def add_person(self, first_name: str, last_name: str) -> int:
        """Insert a new person and return their id."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "INSERT INTO person (first_name, last_name) VALUES (?, ?)",
                (first_name.strip(), last_name.strip()),
            )
            return cursor.lastrowid

    def get_person(self, person_id: int) -> Optional[dict]:
        """Fetch a person by id."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM person WHERE id = ?", (person_id,)
            ).fetchone()
            return dict(row) if row else None

    def update_recognition(self, person_id: int):
        """Increment recognition count and update last_seen_at."""
        with self._get_conn() as conn:
            conn.execute(
                """UPDATE person
                   SET recognition_count = recognition_count + 1,
                       last_seen_at = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (person_id,),
            )

    def get_all_persons(self) -> list[dict]:
        """Return all persons."""
        with self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM person ORDER BY id").fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Embedding CRUD
    # ------------------------------------------------------------------

    def add_embedding(self, person_id: int, embedding: np.ndarray, label: str = "frontal") -> int:
        """Store an embedding for a person."""
        blob = _adapt_array(embedding)
        with self._get_conn() as conn:
            cursor = conn.execute(
                "INSERT INTO face_embedding (person_id, embedding, label) VALUES (?, ?, ?)",
                (person_id, blob, label),
            )
            return cursor.lastrowid

    def get_embeddings_for_person(self, person_id: int) -> list[np.ndarray]:
        """Return all embeddings for a given person."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT embedding FROM face_embedding WHERE person_id = ?",
                (person_id,),
            ).fetchall()
            return [_convert_array(row["embedding"]) for row in rows]

    def get_all_embeddings(self) -> list[tuple[int, int, np.ndarray]]:
        """
        Return ALL embeddings in the database.
        Returns list of (embedding_id, person_id, embedding_vector).
        """
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT id, person_id, embedding FROM face_embedding ORDER BY id"
            ).fetchall()
            return [
                (row["id"], row["person_id"], _convert_array(row["embedding"]))
                for row in rows
            ]

    def get_embedding_count_for_person(self, person_id: int) -> int:
        """How many embeddings does this person have?"""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM face_embedding WHERE person_id = ?",
                (person_id,),
            ).fetchone()
            return row["cnt"]

    def delete_person(self, person_id: int):
        """Delete a person and all their embeddings (CASCADE)."""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM person WHERE id = ?", (person_id,))

    def get_total_embedding_count(self) -> int:
        with self._get_conn() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM face_embedding").fetchone()
            return row["cnt"]