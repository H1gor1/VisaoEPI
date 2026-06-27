from abc import ABC, abstractmethod
from typing import Any

from .types import Embedding


class VectorDb(ABC):

    @abstractmethod
    def search(self, v: Embedding, top_k: int = 1) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def create_or_update(self, v: Embedding, frame_id: int) -> int:
        pass

    @property
    @abstractmethod
    def total_workers(self) -> int:
        pass

    @property
    @abstractmethod
    def total_entries(self) -> int:
        pass


    @abstractmethod
    def get_data(self, worker_id: int, key: str, default: Any | None = None):
        pass

    @abstractmethod
    def update(self, worker_id: int, data: dict[str, Any]):
        pass
