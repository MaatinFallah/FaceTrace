"""
Recognition Manager:
Orchestrates face recognition logic, including:
  - Matching detected faces against the database
  - Cooldown management (avoids counting the same person many times per second)
  - Unknown face confirmation (requires N consecutive frames before prompting)
  - Adding new persons with multiple embeddings
"""

import time
import numpy as np
from typing import Optional
from database import Database
from face_engine import FaceEngine, FaceInfo
from config import (
    RECOGNITION_COOLDOWN_SECONDS,
    UNKNOWN_CONFIRMATION_FRAMES,
)


class PersonSighting:
    """Tracks a recognized person's recent sighting for cooldown."""

    def __init__(self, person_id: int, person_data: dict, similarity: float):
        self.person_id = person_id
        self.person_data = person_data
        self.similarity = similarity
        self.last_counted_at: float = 0.0  # epoch time of last count increment
        self.last_seen_at: float = time.time()

    def should_count(self) -> bool:
        """Check if enough time has passed to count a new recognition."""
        now = time.time()
        return (now - self.last_counted_at) >= RECOGNITION_COOLDOWN_SECONDS

    def mark_counted(self):
        self.last_counted_at = time.time()
        self.last_seen_at = time.time()

    def update_seen(self):
        self.last_seen_at = time.time()


class UnknownFaceTracker:
    """
    Tracks an unknown face across frames to require confirmation
    before prompting to add to database.
    Uses embedding similarity to track the SAME unknown face.
    """

    def __init__(self, embedding: np.ndarray):
        self.embeddings: list[np.ndarray] = [embedding]
        self.consecutive_frames: int = 1
        self.first_seen: float = time.time()
        self.last_seen_at: float = time.time()
        self.confirmed: bool = False
        self.being_registered: bool = False

    def matches(self, embedding: np.ndarray, threshold: float = 0.5) -> bool:
        """Check if a new embedding matches this tracked unknown face."""
        avg_emb = np.mean(self.embeddings[-5:], axis=0)
        avg_emb = avg_emb / np.linalg.norm(avg_emb)
        sim = float(np.dot(avg_emb, embedding))
        return sim >= threshold

    def update(self, embedding: np.ndarray):
        self.embeddings.append(embedding)
        self.consecutive_frames += 1
        self.last_seen_at = time.time()
        if self.consecutive_frames >= UNKNOWN_CONFIRMATION_FRAMES:
            self.confirmed = True

    def get_best_embeddings(self, count: int = 5) -> list[np.ndarray]:
        """
        Return a diverse set of embeddings from the collected ones.
        Picks the most spread-out embeddings for multi-angle coverage.
        """
        if len(self.embeddings) <= count:
            return self.embeddings

        # Greedy farthest-point sampling for diversity
        selected = [self.embeddings[0]]
        used = {0}

        for _ in range(count - 1):
            best_idx = -1
            best_min_dist = -1.0

            for i, emb in enumerate(self.embeddings):
                if i in used:
                    continue
                min_dist = min(
                    float(np.linalg.norm(emb - s)) for s in selected
                )
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = i

            if best_idx >= 0:
                selected.append(self.embeddings[best_idx])
                used.add(best_idx)

        return selected


class RecognitionManager:
    """
    Main orchestrator for face recognition.

    Per frame:
    1. Detect faces
    2. For each face, search the index
    3. If matched -> track sighting, apply cooldown for counting
    4. If unknown -> track across frames, confirm before adding
    """

    def __init__(self):
        self.db = Database()
        self.engine = FaceEngine()

        # Cooldown tracking: person_id -> PersonSighting
        self._active_sightings: dict[int, PersonSighting] = {}

        # Unknown face tracking
        self._unknown_trackers: list[UnknownFaceTracker] = []

        # Flag for when we're in "registration mode"
        self.registration_in_progress: bool = False
        self.pending_tracker: Optional[UnknownFaceTracker] = None

        # Build initial index
        self._rebuild_index()

    def _rebuild_index(self):
        """Load all embeddings from DB and rebuild the Ball Tree."""
        all_embeddings = self.db.get_all_embeddings()
        self.engine.build_index(all_embeddings)

    def process_frame(self, frame: np.ndarray) -> list[dict]:
        """
        Process a single video frame.
        """
        if self.registration_in_progress:
            return self._get_registration_display(frame)

        faces = self.engine.detect_faces(frame)
        results = []

        seen_person_ids = set()
        matched_tracker_indices = set()

        for face_obj in faces:
            face = FaceInfo.from_insightface(face_obj)
            match = self.engine.search(face.embedding)

            if match is not None:
                person_id, similarity = match
                seen_person_ids.add(person_id)

                if person_id in self._active_sightings:
                    sighting = self._active_sightings[person_id]
                    sighting.similarity = similarity
                    sighting.update_seen()
                else:
                    person_data = self.db.get_person(person_id)
                    if person_data is None:
                        continue
                    sighting = PersonSighting(person_id, person_data, similarity)
                    self._active_sightings[person_id] = sighting

                # Apply cooldown before incrementing count in DB
                if sighting.should_count():
                    self.db.update_recognition(person_id)
                    updated = self.db.get_person(person_id)
                    if updated:
                        sighting.person_data = updated
                    sighting.mark_counted()

                results.append({
                    "bbox": face.bbox,
                    "status": "known",
                    "person_data": sighting.person_data,
                    "similarity": similarity,
                    "det_score": face.det_score,
                })

            else:
                # Unknown face
                matched = False
                for i, tracker in enumerate(self._unknown_trackers):
                    if i in matched_tracker_indices:
                        continue
                    if tracker.matches(face.embedding):
                        tracker.update(face.embedding)
                        matched_tracker_indices.add(i)
                        matched = True

                        if tracker.confirmed and not tracker.being_registered:
                            status = "pending"
                        elif tracker.being_registered:
                            status = "registering"
                        else:
                            status = "unknown"

                        results.append({
                            "bbox": face.bbox,
                            "status": status,
                            "tracker": tracker,
                            "frames_seen": tracker.consecutive_frames,
                            "det_score": face.det_score,
                        })
                        break

                if not matched:
                    tracker = UnknownFaceTracker(face.embedding)
                    self._unknown_trackers.append(tracker)
                    results.append({
                        "bbox": face.bbox,
                        "status": "unknown",
                        "tracker": tracker,
                        "frames_seen": 1,
                        "det_score": face.det_score,
                    })

        # Cleanup old unknown trackers (not seen for 2 seconds)
        now = time.time()
        self._unknown_trackers = [
            t for t in self._unknown_trackers
            if (now - t.last_seen_at) < 2.0 or t.being_registered
        ]

        # Cleanup old sightings (not seen for 10 seconds) - FIXED: last_seen_at
        expired = [
            pid for pid, s in self._active_sightings.items()
            if (now - s.last_seen_at) > 10.0
        ]
        for pid in expired:
            del self._active_sightings[pid]

        return results

    def _get_registration_display(self, frame: np.ndarray) -> list[dict]:
        """While registration is in progress, show the face detection box."""
        faces = self.engine.detect_faces(frame)
        results = []
        for face_obj in faces:
            face = FaceInfo.from_insightface(face_obj)
            results.append({
                "bbox": face.bbox,
                "status": "registering",
                "det_score": face.det_score,
            })
        return results

    def start_registration(self, tracker: UnknownFaceTracker):
        """Mark that we are registering this unknown face."""
        tracker.being_registered = True
        self.registration_in_progress = True
        self.pending_tracker = tracker

    def complete_registration(self, first_name: str, last_name: str) -> int:
        """
        Save the new person with multiple embeddings and rebuild index.
        """
        if self.pending_tracker is None:
            raise RuntimeError("No pending registration")

        # 1. Add person to DB
        person_id = self.db.add_person(first_name, last_name)

        # 2. Extract multi-angle embeddings
        embeddings = self.pending_tracker.get_best_embeddings(count=5)
        for i, emb in enumerate(embeddings):
            label = "primary" if i == 0 else f"angle_{i}"
            self.db.add_embedding(person_id, emb, label)

        # 3. Rebuild search index
        self._rebuild_index()

        # 4. Clean up state
        self.registration_in_progress = False
        self.pending_tracker = None
        self._unknown_trackers.clear()

        print(f"[Register] Added {first_name} {last_name} (ID: {person_id}) "
              f"with {len(embeddings)} embeddings.")

        return person_id

    def cancel_registration(self):
        """Cancel an in-progress registration."""
        if self.pending_tracker:
            self.pending_tracker.being_registered = False
        self.registration_in_progress = False
        self.pending_tracker = None

    def get_stats(self) -> dict:
        """Return database and tracking statistics."""
        persons = self.db.get_all_persons()
        total_embeddings = self.db.get_total_embedding_count()
        return {
            "total_persons": len(persons),
            "total_embeddings": total_embeddings,
            "active_sightings": len(self._active_sightings),
            "unknown_trackers": len(self._unknown_trackers),
        }