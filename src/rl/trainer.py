from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import copy
import random
from typing import Callable, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from src.engine import TicTacToeEnv
from src.model import DQNNet, masked_argmax
from src.model.dqn import legal_mask_from_state
from src.rl.buffer import ReplayBuffer, Transition


@dataclass
class TrainerConfig:
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 64
    replay_capacity: int = 10_000
    target_sync_steps: int = 200
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10_000
    device: str = "cpu"


class DQNTrainer:
    def __init__(self, config: TrainerConfig | None = None) -> None:
        self.config = config or TrainerConfig()
        self.device = torch.device(self.config.device)

        self.env = TicTacToeEnv()
        self.online = DQNNet().to(self.device)
        self.target = DQNNet().to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=self.config.lr)
        self.buffer = ReplayBuffer(self.config.replay_capacity)
        self.global_step = 0
        self.last_loss = 0.0
        self.last_val_loss = 0.0

    def to_dict(self) -> Dict:
        return asdict(self.config) | {
            "global_step": self.global_step,
            "buffer_size": len(self.buffer),
            "last_loss": self.last_loss,
            "last_val_loss": self.last_val_loss,
        }

    def apply_train_overrides(
        self,
        *,
        lr: float | None = None,
        batch_size: int | None = None,
        gamma: float | None = None,
        epsilon_start: float | None = None,
        epsilon_end: float | None = None,
        epsilon_decay_steps: int | None = None,
        target_sync_steps: int | None = None,
    ) -> None:
        if lr is not None:
            self.config.lr = float(lr)
            for g in self.optimizer.param_groups:
                g["lr"] = self.config.lr
        if batch_size is not None:
            self.config.batch_size = int(batch_size)
        if gamma is not None:
            self.config.gamma = float(gamma)
        if epsilon_start is not None:
            self.config.epsilon_start = float(epsilon_start)
        if epsilon_end is not None:
            self.config.epsilon_end = float(epsilon_end)
        if epsilon_decay_steps is not None:
            self.config.epsilon_decay_steps = int(epsilon_decay_steps)
        if target_sync_steps is not None:
            self.config.target_sync_steps = int(target_sync_steps)

    def reset_for_new_run(self) -> None:
        # Start each run from a fresh model instead of continuing last weights.
        self.online = DQNNet().to(self.device)
        self.target = DQNNet().to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = optim.Adam(self.online.parameters(), lr=self.config.lr)
        self.buffer = ReplayBuffer(self.config.replay_capacity)
        self.global_step = 0
        self.last_loss = 0.0
        self.last_val_loss = 0.0

    def epsilon(self) -> float:
        progress = min(self.global_step / self.config.epsilon_decay_steps, 1.0)
        return self.config.epsilon_start + progress * (self.config.epsilon_end - self.config.epsilon_start)

    def select_action(self, state_np: np.ndarray, training: bool = True) -> int:
        legal = [i for i, v in enumerate(state_np.reshape(-1)) if v == 0.0]
        if not legal:
            return 0
        if training and random.random() < self.epsilon():
            return random.choice(legal)

        with torch.no_grad():
            state = torch.tensor(state_np, dtype=torch.float32, device=self.device)
            q = self.online(state)
            mask = legal_mask_from_state(state)
            action = masked_argmax(q, mask).item()
        return int(action)

    def _opponent_move(self, state_np: np.ndarray) -> int:
        legal = [i for i, v in enumerate(state_np.reshape(-1)) if v == 0.0]
        return random.choice(legal) if legal else 0

    def play_one_episode(self) -> float:
        state = self.env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action = self.select_action(state, training=True)
            step = self.env.step(action, player=1)
            reward = step.reward
            done = step.done
            next_state = step.state

            if not done:
                opp_action = self._opponent_move(next_state)
                opp_step = self.env.step(opp_action, player=-1)
                next_state = opp_step.state
                if opp_step.done:
                    done = True
                    if opp_step.winner == -1:
                        reward = -1.0
                    elif opp_step.winner == 0:
                        reward = 0.5

            self.buffer.push(
                Transition(
                    state=state.astype(np.float32),
                    action=int(action),
                    reward=float(reward),
                    next_state=next_state.astype(np.float32),
                    done=bool(done),
                )
            )

            state = next_state
            episode_reward += reward
            self.global_step += 1

            if len(self.buffer) >= self.config.batch_size:
                self.last_loss = float(self.train_step())
            if self.global_step % self.config.target_sync_steps == 0:
                self.target.load_state_dict(self.online.state_dict())

        return episode_reward

    def train_step(self) -> float:
        batch = self.buffer.sample(self.config.batch_size)

        states = torch.tensor(np.vstack([t.state for t in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.vstack([t.next_state for t in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.online(states).gather(1, actions)

        with torch.no_grad():
            next_q = self.target(next_states)
            next_mask = legal_mask_from_state(next_states)
            next_actions = masked_argmax(next_q, next_mask).unsqueeze(1)
            next_best_q = next_q.gather(1, next_actions)
            target_q = rewards + (1.0 - dones) * self.config.gamma * next_best_q

        loss = F.mse_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validation_step(self) -> float:
        if len(self.buffer) < self.config.batch_size:
            return self.last_loss
        batch = self.buffer.sample(self.config.batch_size)
        states = torch.tensor(np.vstack([t.state for t in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.vstack([t.next_state for t in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device).unsqueeze(1)

        with torch.no_grad():
            q_values = self.online(states).gather(1, actions)
            next_q = self.target(next_states)
            next_mask = legal_mask_from_state(next_states)
            next_actions = masked_argmax(next_q, next_mask).unsqueeze(1)
            next_best_q = next_q.gather(1, next_actions)
            target_q = rewards + (1.0 - dones) * self.config.gamma * next_best_q
            loss = F.mse_loss(q_values, target_q)
        return float(loss.item())

    def train(self, episodes: int = 200, on_episode_end: Callable[[Dict], None] | None = None) -> List[Dict]:
        logs: List[Dict] = []
        for ep in range(1, episodes + 1):
            reward = self.play_one_episode()
            item = {
                "episode": ep,
                "reward": reward,
                "epsilon": self.epsilon(),
                "loss": self.last_loss,
                "val_loss": self.validation_step(),
            }
            self.last_val_loss = float(item["val_loss"])
            logs.append(item)
            if on_episode_end is not None:
                on_episode_end(item)
        return logs

    def snapshot_state(self) -> Dict:
        return {
            "online_state_dict": copy.deepcopy(self.online.state_dict()),
            "target_state_dict": copy.deepcopy(self.target.state_dict()),
            "optimizer_state_dict": copy.deepcopy(self.optimizer.state_dict()),
            "global_step": self.global_step,
            "last_loss": self.last_loss,
            "last_val_loss": self.last_val_loss,
            "config": asdict(self.config),
        }

    def restore_snapshot(self, snapshot: Dict) -> None:
        self.online.load_state_dict(snapshot["online_state_dict"])
        self.target.load_state_dict(snapshot["target_state_dict"])
        if "optimizer_state_dict" in snapshot:
            self.optimizer.load_state_dict(snapshot["optimizer_state_dict"])
        self.global_step = int(snapshot.get("global_step", self.global_step))
        self.last_loss = float(snapshot.get("last_loss", self.last_loss))
        self.last_val_loss = float(snapshot.get("last_val_loss", self.last_val_loss))

    def save_checkpoint(self, path: str, snapshot: Dict | None = None) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        state = snapshot or self.snapshot_state()
        torch.save(
            state,
            p,
        )

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online_state_dict"])
        self.target.load_state_dict(ckpt["target_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = int(ckpt.get("global_step", self.global_step))
        self.last_loss = float(ckpt.get("last_loss", self.last_loss))
        self.last_val_loss = float(ckpt.get("last_val_loss", self.last_val_loss))
