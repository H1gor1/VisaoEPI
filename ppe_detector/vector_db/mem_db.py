from typing import Any, DefaultDict

import numpy as np

import faiss

from .base import VectorDb
from .types import Embedding


class MemVecDb(VectorDb):

    def __init__(self, dim: int = 512, threshold: float = 0.8):
        self.index = faiss.IndexFlatL2(dim)
        self.metadata: list[dict[str, Any]] = []
        self.next_worker_id = 0
        self.threshold = threshold
        self.history: DefaultDict[int, list[int]] = DefaultDict(lambda: [])
        self._worker_id_data: DefaultDict[int, dict[str, Any]] = DefaultDict(
            lambda: {}
        )

    def search(
        self,
        v: Embedding,
        top_k: int = 1,
    ) -> list[dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        distances, indices = self.index.search(
            np.array([v], dtype=np.float32),
            top_k,
        )

        return [
            {
                "distance": float(distance),
                "metadata": self.metadata[index],
            }
            for distance, index in zip(distances[0], indices[0])
            if index != -1
        ]

    def create_or_update(
        self,
        v: Embedding,
        frame_id: int,
    ) -> int:
        results = self.search(v)

        if results and results[0]["distance"] < self.threshold:
            worker_id = results[0]["metadata"]["worker_id"]
        else:
            worker_id = self.next_worker_id
            self.next_worker_id += 1

        self.index.add(np.array([v], dtype=np.float32))

        self.metadata.append(
            {
                "worker_id": worker_id,
                "frame_id": frame_id,
            }
        )

        self.history[worker_id].append(frame_id)

        return worker_id

    def total_workers(self) -> int:
        return self.next_worker_id

    def total_entries(self) -> int:
        return len(self.metadata)

    def get_data(self, worker_id: int, key: str, default: Any | None = None):
        return self._worker_id_data[worker_id].get(key, default)

    def update(self, worker_id: int, data: dict[str, Any]):
        self._worker_id_data[worker_id].update(data)

    def get_all_data(self, worker_id: int) -> dict[str, Any]:
        return self._worker_id_data[worker_id]
