"""
Face detection and embedding extraction using InsightFace with GPU acceleration.
"""

import numpy as np
import cv2
import onnxruntime as ort
from insightface.app import FaceAnalysis
from sklearn.neighbors import BallTree
from typing import Optional
from config import SIMILARITY_THRESHOLD, ANN_CANDIDATES


class FaceEngine:
    def __init__(self):
        # 1. Determine the best available provider
        available = ort.get_available_providers()
        providers = []

        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        if "DmlExecutionProvider" in available:  # DirectML (Windows GPU)
            providers.append("DmlExecutionProvider")
        providers.append("CPUExecutionProvider")

        print(f"[FaceEngine] Initializing with providers: {providers}")

        # 2. Initialize InsightFace
        # det_size=(320, 320) or (480, 480) is 3x faster than (640, 640)
        # ctx_id=0 uses the first GPU device.
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=providers,
        )
        self.app.prepare(ctx_id=0, det_size=(320, 320))

        # 3. Search index
        self._tree: Optional[BallTree] = None
        self._index_person_ids: np.ndarray = np.array([], dtype=np.int64)
        self._index_embedding_ids: np.ndarray = np.array([], dtype=np.int64)
        self._index_embeddings: Optional[np.ndarray] = None

    def detect_faces(self, frame: np.ndarray) -> list:
        faces = self.app.get(frame)
        return [f for f in faces if f.det_score >= 0.5]

    @staticmethod
    def get_embedding(face) -> np.ndarray:
        emb = face.embedding
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb.astype(np.float32)

    def build_index(self, embeddings_data: list[tuple[int, int, np.ndarray]]):
        if not embeddings_data:
            self._tree = None
            self._index_person_ids = np.array([], dtype=np.int64)
            self._index_embedding_ids = np.array([], dtype=np.int64)
            self._index_embeddings = None
            return

        embedding_ids = np.array([e[0] for e in embeddings_data], dtype=np.int64)
        person_ids = np.array([e[1] for e in embeddings_data], dtype=np.int64)
        vectors = np.array([e[2] for e in embeddings_data], dtype=np.float32)

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors = vectors / norms

        self._tree = BallTree(vectors, leaf_size=40, metric="euclidean")
        self._index_person_ids = person_ids
        self._index_embedding_ids = embedding_ids
        self._index_embeddings = vectors

        print(f"[Index] Built Ball Tree with {len(vectors)} embeddings for {len(set(person_ids))} persons")

    def search(self, query_embedding: np.ndarray) -> Optional[tuple[int, float]]:
        if self._tree is None or len(self._index_person_ids) == 0:
            return None

        query = query_embedding.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        k = min(ANN_CANDIDATES, len(self._index_person_ids))
        distances, indices = self._tree.query(query, k=k)

        best_person_id = None
        best_similarity = -1.0

        for dist, idx in zip(distances[0], indices[0]):
            cosine_sim = 1.0 - (dist ** 2) / 2.0
            if cosine_sim > best_similarity:
                best_similarity = cosine_sim
                best_person_id = int(self._index_person_ids[idx])

        if best_similarity >= SIMILARITY_THRESHOLD:
            return (best_person_id, float(best_similarity))

        return None


class FaceInfo:
    def __init__(self, bbox, embedding, det_score, pose=None):
        self.bbox = bbox
        self.embedding = embedding
        self.det_score = det_score
        self.pose = pose

    @classmethod
    def from_insightface(cls, face) -> "FaceInfo":
        emb = FaceEngine.get_embedding(face)
        pose = getattr(face, "pose", None)
        return cls(
            bbox=face.bbox.astype(int),
            embedding=emb,
            det_score=float(face.det_score),
            pose=pose,
        )