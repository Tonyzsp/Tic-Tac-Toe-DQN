from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Deque, List

import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 10_000) -> None:
        self.capacity = capacity
        self._data: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._data)

    def push(self, item: Transition) -> None:
        self._data.append(item)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self._data, k=batch_size)
