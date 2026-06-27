from dataclasses import dataclass
from typing import Self, NamedTuple

import config


class Coords(NamedTuple):
    x1: int
    y1: int
    x2: int
    y2: int


TOLERANCE = 20


@dataclass(frozen=True)
class Detection:
    cls_id: int
    cls_name: str
    confidence: float
    bbox: Coords

    def contains(self, other: Self) -> bool:
        x1s, y1s, x2s, y2s = self.bbox
        x1o, y1o, x2o, y2o = other.bbox
        return (
            (x1o + TOLERANCE >= x1s)
            and y1o + TOLERANCE >= y1s
            and x2o - TOLERANCE <= x2s
            and y2o - TOLERANCE <= y2s
        )

    def is_violation(self) -> bool:
        return self.cls_id in config.VIOLATION_CLASSES
