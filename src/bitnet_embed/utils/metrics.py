from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RunningAverage:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def value(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


@dataclass(slots=True)
class ThroughputMeter:
    items: int = 0
    seconds: float = 0.0
    history: list[float] = field(default_factory=list)

    def update(self, items: int, seconds: float) -> None:
        self.items += items
        self.seconds += seconds
        if seconds > 0:
            self.history.append(items / seconds)

    @property
    def per_second(self) -> float:
        if self.seconds <= 0:
            return 0.0
        return self.items / self.seconds
