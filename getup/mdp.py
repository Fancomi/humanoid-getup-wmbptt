"""Analytic, differentiable MDP heads for get-up (no reference motion, no FK -> robot agnostic).

State tuple ``S = (p, R, v, w, q, qd)``: root position, rotation matrix, world linear/angular
velocity, joint positions and velocities. Flat layout: p(3) quat(4) v(3) w(3) q(nj) qd(nj).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .rotations import mat_to_6d, mat_to_quat, mtm, mtv, quat_to_mat, yaw_mat


def flatten_state(S) -> torch.Tensor:
    p, R, v, w, q, qd = S
    return torch.cat((p, mat_to_quat(R), v, w, q, qd), -1)


def unflatten_state(x: torch.Tensor, nj: int):
    return x[..., 0:3], quat_to_mat(x[..., 3:7]), x[..., 7:10], x[..., 10:13], x[..., 13 : 13 + nj], x[..., 13 + nj : 13 + 2 * nj]


def invariant_features(S) -> torch.Tensor:
    """Heading/xy-invariant: height, tilt (6D), body-frame velocities, joints."""
    p, R, v, w, q, qd = S
    H = yaw_mat(R)
    return torch.cat(
        (p[..., 2:3], mat_to_6d(mtm(H, R)), mtv(R, v), mtv(R, w), q, qd), -1
    )


def inv_dim(nj: int) -> int:
    return 1 + 6 + 3 + 3 + 2 * nj


@dataclass
class RewardCfg:
    w_up: float = 1.0
    w_height: float = 2.0
    w_pose: float = 1.0
    w_still: float = 0.5
    w_action_rate: float = -0.02
    w_qd: float = -1e-4
    w_joint_limit: float = -2.0
    soft_limit_factor: float = 0.95
    pose_std2: float = 0.1
    stand_frac: float = 0.85  # "standing" gate: root height above this fraction of the standing height


class GetupMDP:
    def __init__(self, stand_q: torch.Tensor, stand_h: float, q_lo, q_hi, action_scale: float, dt: float, cfg: RewardCfg = RewardCfg()):
        self.stand_q, self.stand_h, self.cfg, self.dt = stand_q, stand_h, cfg, dt
        self.nj = stand_q.shape[-1]
        self.q_lo, self.q_hi = q_lo, q_hi
        mid, half = (q_lo + q_hi) / 2, (q_hi - q_lo) / 2
        self.soft_lo, self.soft_hi = mid - half * cfg.soft_limit_factor, mid + half * cfg.soft_limit_factor
        self.alpha = action_scale
        # action space bounds that map to the joint ranges (targets are clipped to the actuator range)
        self.a_lo, self.a_hi = (q_lo - stand_q) / action_scale, (q_hi - stand_q) / action_scale
        self.obs_dim = 3 + 3 + 3 * self.nj
        self.max_step_reward = (cfg.w_up + cfg.w_height + cfg.w_pose + cfg.w_still) * dt

    def clip_action(self, a: torch.Tensor) -> torch.Tensor:
        return torch.maximum(torch.minimum(a, self.a_hi), self.a_lo)

    def target(self, a: torch.Tensor) -> torch.Tensor:
        return self.stand_q + self.alpha * a

    # ---------------- observation (proprioceptive: what an IMU + encoders would give) ----------------
    def obs(self, S, a_prev) -> torch.Tensor:
        p, R, v, w, q, qd = S
        g_b = -R[..., 2, :]
        w_b = mtv(R, w)
        return torch.cat((g_b, 0.25 * w_b, q - self.stand_q, 0.1 * qd, a_prev), -1)

    def add_noise(self, o: torch.Tensor, scale: float = 1.0, generator: torch.Generator | None = None) -> torch.Tensor:
        n = torch.full_like(o[:1], 0.01)
        n[..., 0:3] = 0.03
        n[..., 3:6] = 0.05
        u = torch.rand(o.shape, device=o.device, generator=generator) if generator is not None else torch.rand_like(o)
        return o + scale * (2 * u - 1) * n

    # ---------------- reward ----------------
    def reward(self, S, a, a_prev, terms: bool = False):
        c = self.cfg
        p, R, v, w, q, qd = S
        up = R[..., 2, 2]
        h = p[..., 2]
        r_up = 0.5 * (1 + up)
        r_h = torch.clamp(h / self.stand_h, max=1.0)
        standing = torch.sigmoid((h - c.stand_frac * self.stand_h) / 0.02) * torch.sigmoid((up - 0.9) / 0.02)
        r_pose = torch.exp(-((q - self.stand_q) ** 2).mean(-1) / c.pose_std2)
        r_still = torch.exp(-((v[..., :2] ** 2).sum(-1) + 0.1 * (w**2).sum(-1)))
        r_rate = ((a - a_prev) ** 2).sum(-1)
        r_qd = (qd**2).sum(-1)
        r_lim = (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1)
        r = (c.w_up * r_up + c.w_height * r_h + standing * (c.w_pose * r_pose + c.w_still * r_still)
             + c.w_action_rate * r_rate + c.w_qd * r_qd + c.w_joint_limit * r_lim) * self.dt  # fmt: skip
        if terms:
            return r, {"up": up, "height": h, "standing": standing, "pose": r_pose, "action_rate": r_rate}
        return r

    def critic_features(self, S) -> torch.Tensor:
        p, R, v, w, q, qd = S
        return torch.cat((invariant_features(S), q - self.stand_q), -1)

    def critic_dim(self) -> int:
        return inv_dim(self.nj) + self.nj
