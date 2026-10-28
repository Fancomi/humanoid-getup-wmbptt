"""Replay buffer of simulator transitions laid out as (time, env), stored on the training device."""

from __future__ import annotations

import torch


class ReplayBuffer:
    def __init__(self, T: int, N: int, state_dim: int, nj: int, device: str):
        self.T, self.N, self.device = T, N, device
        f = dict(device=device)
        self.data = {
            "s": torch.zeros(T, N, state_dim, **f),
            "s1": torch.zeros(T, N, state_dim, **f),
            "a": torch.zeros(T, N, nj, **f),
            "a_prev": torch.zeros(T, N, nj, **f),
            "ep_step": torch.zeros(T, N, dtype=torch.long, **f),
            "done": torch.zeros(T, N, dtype=torch.bool, **f),
        }
        self.ptr, self.size = 0, 0
        # device copies so sampling needs no host sync and can be captured in a CUDA graph
        self.ptr_t = torch.zeros((), dtype=torch.long, device=device)
        self.size_t = torch.zeros((), dtype=torch.long, device=device)

    def add(self, tr: dict):
        for k, v in self.data.items():
            v[self.ptr] = tr[k]
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)
        self.ptr_t.fill_(self.ptr)
        self.size_t.fill_(self.size)

    def _row(self, age):
        return (self.ptr_t - self.size_t + age) % self.T

    def _uniform_int(self, B: int, n):
        """B ints uniform in [0, n) for a device tensor n."""
        return torch.minimum((torch.rand(B, device=self.device) * n).long(), n - 1)

    def sample_windows(self, B: int, k: int):
        t0 = self._uniform_int(B, self.size_t - k + 1)
        env = torch.randint(0, self.N, (B,), device=self.device)
        rows = self._row(t0.unsqueeze(-1) + torch.arange(k, device=self.device))
        e = env.unsqueeze(-1).expand_as(rows)
        out = {key: self.data[key][rows, e] for key in ("s", "s1", "a", "done")}
        d = out["done"].float()
        out["mask"] = torch.cumprod(torch.cat((torch.ones(B, 1, device=self.device), 1 - d[:, :-1]), -1), -1)
        return out

    def sample_starts(self, B: int, hist: int):
        age = self._uniform_int(B, self.size_t)
        env = torch.randint(0, self.N, (B,), device=self.device)
        row = self._row(age)
        ep_step = self.data["ep_step"][row, env]
        back = torch.arange(hist - 1, -1, -1, device=self.device)
        back = torch.minimum(back.unsqueeze(0), ep_step.unsqueeze(-1))
        back = torch.minimum(back, age.unsqueeze(-1))
        hrows = self._row(age.unsqueeze(-1) - back)
        henv = env.unsqueeze(-1).expand_as(hrows)
        start = {k: self.data[k][row, env] for k in ("s", "a_prev")}
        hist_d = {k: self.data[k][hrows, henv] for k in ("s", "a_prev")}
        return start, hist_d
