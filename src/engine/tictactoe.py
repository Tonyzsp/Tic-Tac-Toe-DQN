from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


WIN_LINES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


@dataclass
class StepResult:
    state: np.ndarray
    reward: float
    done: bool
    illegal_move: bool
    winner: int


class TicTacToeEnv:
    """Simple environment with +1 agent and -1 opponent."""

    def __init__(self) -> None:
        self.board = np.zeros(9, dtype=np.float32)
        self.done = False
        self.winner = 0

    def reset(self) -> np.ndarray:
        self.board[:] = 0.0
        self.done = False
        self.winner = 0
        return self.state

    @property
    def state(self) -> np.ndarray:
        return self.board.copy().reshape(1, 9)

    def legal_actions(self) -> List[int]:
        return [i for i, v in enumerate(self.board) if v == 0.0]

    def _check_winner(self) -> int:
        for a, b, c in WIN_LINES:
            s = self.board[a] + self.board[b] + self.board[c]
            if s == 3:
                return 1
            if s == -3:
                return -1
        return 0

    def _is_draw(self) -> bool:
        return bool(np.all(self.board != 0.0) and self._check_winner() == 0)

    def _terminal_result(self) -> Tuple[bool, int]:
        winner = self._check_winner()
        if winner != 0:
            return True, winner
        if self._is_draw():
            return True, 0
        return False, 0

    def step(self, action: int, player: int = 1) -> StepResult:
        if self.done:
            return StepResult(self.state, 0.0, True, False, self.winner)

        if action < 0 or action >= 9 or self.board[action] != 0.0:
            self.done = True
            self.winner = -player
            return StepResult(self.state, -10.0, True, True, self.winner)

        self.board[action] = float(player)
        done, winner = self._terminal_result()
        self.done = done
        self.winner = winner

        if done:
            if winner == player:
                reward = 1.0
            elif winner == 0:
                reward = 0.5
            else:
                reward = -1.0
        else:
            reward = 0.0

        return StepResult(self.state, reward, done, False, winner)
