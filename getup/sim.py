"""Batched MuJoCo simulator on CPU threads (mujoco.rollout), with fallen-state resets.

State layout handed to torch: p(3), quat wxyz(4), v_world(3), w_body->world(3), q(nj), qd(nj).
MuJoCo free-joint qvel stores linear velocity in world frame and angular velocity in the body frame;
we convert angular velocity to world frame so the state convention matches SuperMimic.
"""

from __future__ import annotations

import os

import mujoco
import numpy as np
from mujoco import rollout

from .robots import RobotCfg


def _quat_to_mat_np(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
         2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
         2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1,
    ).reshape(q.shape[:-1] + (3, 3))  # fmt: skip


class BatchSim:
    def __init__(self, cfg: RobotCfg, num_envs: int, nthread: int | None = None, seed: int = 0):
        self.cfg = cfg
        self.m = mujoco.MjModel.from_xml_path(cfg.xml)
        if cfg.sim_dt:
            self.m.opt.timestep = cfg.sim_dt
        if cfg.solver_iterations:
            self.m.opt.iterations = cfg.solver_iterations
        self.substeps = int(round(cfg.control_dt / self.m.opt.timestep))
        self.N = num_envs
        self.nj = self.m.nu
        # sched_getaffinity respects container/cgroup CPU pinning on Linux; macOS lacks it
        self.nthread = nthread or (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count())
        self.datas = [mujoco.MjData(self.m) for _ in range(self.nthread)]
        self.pool = rollout.Rollout(nthread=self.nthread)
        self.spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        self.nstate = mujoco.mj_stateSize(self.m, self.spec)
        self.rng = np.random.default_rng(seed)
        key = self.m.key(cfg.stand_key)
        self.stand_qpos = key.qpos.copy()
        self.stand_q = key.qpos[7:].copy()
        jr = self.m.jnt_range[1:]  # skip free joint
        self.q_lo, self.q_hi = jr[:, 0].copy(), jr[:, 1].copy()
        self.ctrl_lo, self.ctrl_hi = self.m.actuator_ctrlrange[:, 0].copy(), self.m.actuator_ctrlrange[:, 1].copy()
        self.states = np.zeros((self.N, self.nstate))
        self.warm = np.zeros((self.N, self.m.nv))
        self.fallen_bank: np.ndarray | None = None

    # ---------------------------------------------------------------- state io
    def _full_from_qpos_qvel(self, qpos, qvel):
        d = self.datas[0]
        out = np.zeros((len(qpos), self.nstate))
        for i in range(len(qpos)):
            mujoco.mj_resetData(self.m, d)
            d.qpos[:] = qpos[i]
            d.qvel[:] = qvel[i]
            mujoco.mj_getState(self.m, d, out[i], self.spec)
        return out

    def split(self, states: np.ndarray):
        """FULLPHYSICS state -> (qpos, qvel). Layout: time(1), qpos(nq), qvel(nv), act, ..."""
        nq, nv = self.m.nq, self.m.nv
        return states[..., 1 : 1 + nq], states[..., 1 + nq : 1 + nq + nv]

    def obs_state(self, states: np.ndarray | None = None) -> np.ndarray:
        """(N, 13 + 2 nj) float32: p, quat, v_world, w_world, q, qd."""
        qpos, qvel = self.split(self.states if states is None else states)
        R = _quat_to_mat_np(qpos[..., 3:7])
        w_world = (R @ qvel[..., 3:6, None])[..., 0]
        return np.concatenate((qpos[..., :7], qvel[..., :3], w_world, qpos[..., 7:], qvel[..., 6:]), -1).astype(np.float32)

    # ---------------------------------------------------------------- stepping
    def step(self, targets: np.ndarray, ids: np.ndarray | None = None):
        """Hold joint position targets (N, nj) for one control period."""
        ctrl = np.clip(targets, self.ctrl_lo, self.ctrl_hi)
        st, _ = self.pool.rollout(
            self.m, self.datas, self.states, ctrl[:, None, :].repeat(self.substeps, 1), nstep=self.substeps,
            initial_warmstart=self.warm,
        )
        self.states = st[:, -1].copy()
        return self.states

    # ---------------------------------------------------------------- fallen states
    def build_fallen_bank(self, n: int, settle_s: float = 2.0):
        """Drop the robot from random orientations with random limp joint targets and let it settle."""
        nq = self.m.nq
        qpos = np.tile(self.stand_qpos, (n, 1))
        qvel = np.zeros((n, self.m.nv))
        u = self.rng.normal(size=(n, 4))
        qpos[:, 3:7] = u / np.linalg.norm(u, axis=1, keepdims=True)
        qpos[:, 2] = 0.9
        qpos[:, 7:] = self.rng.uniform(self.q_lo, self.q_hi, (n, self.nj))
        qpos[:, 7:] = 0.5 * qpos[:, 7:] + 0.5 * self.stand_q
        qvel[:, :6] = self.rng.normal(scale=0.5, size=(n, 6))
        init = self._full_from_qpos_qvel(qpos, qvel)
        ctrl = np.clip(qpos[:, 7:], self.ctrl_lo, self.ctrl_hi)
        steps = int(settle_s / self.m.opt.timestep)
        # weak drives while settling: scale gains down so the robot slumps on the floor
        m_soft = mujoco.MjModel.from_xml_path(self.cfg.xml)
        m_soft.opt.timestep, m_soft.opt.iterations = self.m.opt.timestep, self.m.opt.iterations
        m_soft.actuator_gainprm[:, 0] *= 0.1
        m_soft.actuator_biasprm[:, 1:3] *= 0.1
        datas = [mujoco.MjData(m_soft) for _ in range(self.nthread)]
        st, _ = self.pool.rollout(m_soft, datas, init, ctrl[:, None, :], nstep=steps)
        final = st[:, -1].copy()
        final[:, 0] = 0.0  # time
        qp, _ = self.split(final)
        ok = np.isfinite(final).all(1) & (qp[:, 2] < 0.6 * self.cfg.stand_height)
        self.fallen_bank = final[ok]
        self.fallen_ctrl = ctrl[ok]
        return self.fallen_bank

    def reset(self, ids: np.ndarray):
        if len(ids) == 0:
            return
        j = self.rng.integers(0, len(self.fallen_bank), len(ids))
        s = self.fallen_bank[j].copy()
        # small random joint perturbation, zero velocity, random yaw is already in the bank
        nq = self.m.nq
        s[:, 1 + 7 : 1 + nq] = np.clip(s[:, 1 + 7 : 1 + nq] + self.rng.uniform(-0.05, 0.05, (len(ids), self.nj)), self.q_lo, self.q_hi)
        s[:, 1 + nq : 1 + nq + self.m.nv] = 0.0
        self.states[ids] = s
        self.warm[ids] = 0.0
