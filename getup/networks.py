"""World model, policy and critic (same structure as SuperMimic, sizes reduced for a laptop)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .mdp import invariant_features, inv_dim
from .rotations import mm, mtm, mtv, mv, so3_exp, so3_log, yaw_mat

AMP_DTYPE: torch.dtype | None = None


def mlp(inp: int, hidden: list[int], out: int, act=nn.ELU, layer_norm: bool = False) -> nn.Sequential:
    layers, d = [], inp
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act())
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


def run_mlp(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if AMP_DTYPE is None:
        return net(x)
    with torch.autocast(x.device.type, dtype=AMP_DTYPE, cache_enabled=False):  # no weight-cast cache: CUDA-graph safe
        return net(x).float()


class RunningNorm(nn.Module):
    def __init__(self, dim: int, max_count: float = 5e7, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count, self.eps = max_count, eps

    @torch.no_grad()
    def update(self, x: torch.Tensor, w: torch.Tensor | None = None):
        """Branch-free (no host sync, CUDA-graph safe); optional per-row 0/1 weights ``w``; stops updating once
        count reaches max_count."""
        x = x.reshape(-1, x.shape[-1]).float()
        if w is None:
            n = x.shape[0]
            m, v = x.mean(0), x.var(0, unbiased=False)
        else:
            w = w.reshape(-1, 1).float()
            n = w.sum()
            m = (w * x).sum(0) / n.clamp(min=1)
            v = (w * (x - m) ** 2).sum(0) / n.clamp(min=1)
        n_eff = n * (self.count < self.max_count).float()
        tot = self.count + n_eff
        f = n_eff / tot.clamp(min=1)
        delta = m - self.mean
        self.var.copy_(self.var * (1 - f) + v * f + delta**2 * f * (1 - f))
        self.mean.add_(delta * f)
        self.count.copy_(tot)

    def forward(self, x):
        return (x - self.mean) / torch.sqrt(self.var + self.eps**2)

    def std(self):
        return torch.sqrt(self.var + self.eps**2)


class WorldModel(nn.Module):
    """Deterministic s_{t+1} = f(s_t, a_t), predicting frame-local state increments."""

    def __init__(self, nj: int, hidden=(1024, 1024, 1024)):
        super().__init__()
        self.nj = nj
        self.in_dim = inv_dim(nj) + nj
        self.out_dim = 12 + 2 * nj
        self.net = mlp(self.in_dim, list(hidden), self.out_dim, act=nn.SiLU, layer_norm=True)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(self.out_dim, eps=1e-4)

    def inputs(self, S, a):
        return torch.cat((invariant_features(S), a), -1)

    @staticmethod
    def targets(S, S1):
        p, R, v, w, q, qd = S
        p1, R1, v1, w1, q1, qd1 = S1
        H = yaw_mat(R)
        return torch.cat(
            (mtv(H, p1 - p), so3_log(mtm(R, R1)), mtv(R, v1 - v), mtv(R, w1 - w), q1 - q, qd1 - qd), -1,
        )  # fmt: skip

    state_error = targets  # error of S_hat w.r.t. S in the same frame-local layout: targets(S, S_hat)

    def step(self, S, a):
        p, R, v, w, q, qd = S
        y = run_mlp(self.net, self.in_norm(self.inputs(S, a)))
        d = y * self.out_norm.std() + self.out_norm.mean
        H = yaw_mat(R)
        dp, dth, dv, dw, dq, dqd = d.split((3, 3, 3, 3, self.nj, self.nj), -1)
        return (
            p + mv(H, dp), mm(R, so3_exp(dth)), v + mv(R, dv), w + mv(R, dw), q + dq, qd + dqd,
        )  # fmt: skip


class Policy(nn.Module):
    def __init__(self, obs_dim: int, nj: int, hist: int, hidden=(512, 256, 128)):
        super().__init__()
        self.hist = hist
        self.net = mlp(obs_dim * hist, list(hidden), nj)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()
        self.obs_norm = RunningNorm(obs_dim)

    def forward(self, obs_hist: torch.Tensor) -> torch.Tensor:
        return run_mlp(self.net, self.obs_norm(obs_hist).flatten(-2))


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = mlp(in_dim, [256, 128], 1)
        self.norm = RunningNorm(in_dim)

    def forward(self, x):
        return run_mlp(self.net, self.norm(x)).squeeze(-1)
