"""Heuristic matching and optional online DQN scheduling."""

from __future__ import annotations

import random
from collections.abc import Sequence

from .features import extract_task_features
from .models import GraphTask, ResourceProfile, ScheduleDecision


class HeuristicScheduler:
    """Choose the available resource with the lowest estimated latency."""

    def schedule(
        self, task: GraphTask, resources: Sequence[ResourceProfile]
    ) -> ScheduleDecision:
        candidates = [
            (resource.estimated_latency_ms(task), resource)
            for resource in resources
            if resource.available
            and task.estimated_memory_bytes <= resource.memory_bytes
        ]
        if not candidates:
            raise RuntimeError(f"no available resource can run task {task.task_id}")
        latency, selected = min(candidates, key=lambda pair: pair[0])
        scores = {
            resource.name: resource.estimated_latency_ms(task)
            for resource in resources
            if resource.available
        }
        return ScheduleDecision(
            task_id=task.task_id,
            device=selected.device,
            resource_name=selected.name,
            estimated_latency_ms=latency,
            scores=scores,
        )


class DQNScheduler:
    """Small online DQN policy; importing this class does not require PyTorch."""

    def __init__(
        self,
        resource_count: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        seed: int = 0,
    ) -> None:
        if resource_count <= 0:
            raise ValueError("resource_count must be positive")
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise RuntimeError(
                "The DQN policy requires PyTorch. Install it with `pip install torch`."
            ) from exc

        self._torch = torch
        self._resource_count = resource_count
        self.gamma = gamma
        self.epsilon = epsilon
        self._random = random.Random(seed)
        self._state_size: int | None = None
        self._online = None
        self._target = None
        self._optimizer = None
        self._loss = nn.MSELoss()
        self._learning_rate = learning_rate

    def _initialize(self, state_size: int) -> None:
        torch = self._torch
        nn = torch.nn
        self._state_size = state_size
        self._online = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self._resource_count),
        )
        self._target = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self._resource_count),
        )
        self._target.load_state_dict(self._online.state_dict())
        self._optimizer = torch.optim.Adam(
            self._online.parameters(), lr=self._learning_rate
        )

    def select_action(
        self, state: Sequence[float], valid_actions: Sequence[int]
    ) -> int:
        valid = tuple(valid_actions)
        if not valid:
            raise ValueError("valid_actions must not be empty")
        if any(action < 0 or action >= self._resource_count for action in valid):
            raise ValueError("valid action is outside the configured resource range")
        if self._state_size is None:
            self._initialize(len(state))
        if len(state) != self._state_size:
            raise ValueError("state vector size changed after DQN initialization")
        if self._random.random() < self.epsilon:
            return self._random.choice(valid)
        with self._torch.no_grad():
            values = self._online(self._torch.tensor(state, dtype=self._torch.float32))
        return max(valid, key=lambda action: float(values[action].item()))

    def observe(
        self,
        state: Sequence[float],
        action: int,
        reward: float,
        next_state: Sequence[float],
        done: bool,
    ) -> float:
        if not 0 <= action < self._resource_count:
            raise ValueError("action is outside the configured resource range")
        if self._state_size is None:
            self._initialize(len(state))
        if len(state) != self._state_size or len(next_state) != self._state_size:
            raise ValueError("state vector size does not match the initialized DQN")

        torch = self._torch
        state_tensor = torch.tensor(state, dtype=torch.float32)
        next_tensor = torch.tensor(next_state, dtype=torch.float32)
        prediction = self._online(state_tensor)[action]
        with torch.no_grad():
            future = self._target(next_tensor).max()
            target = torch.tensor(reward, dtype=torch.float32)
            if not done:
                target = target + self.gamma * future
        loss = self._loss(prediction, target)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self._target.load_state_dict(self._online.state_dict())
        return float(loss.item())

    @staticmethod
    def encode_state(
        task: GraphTask, resources: Sequence[ResourceProfile]
    ) -> tuple[float, ...]:
        values = list(extract_task_features(task).as_vector())
        for resource in resources:
            values.extend(
                (
                    float(resource.available),
                    resource.compute_gflops,
                    resource.bandwidth_gbps,
                    resource.memory_bytes / 1_000_000_000,
                    float(resource.queue_depth),
                )
            )
        return tuple(values)
