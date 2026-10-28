"""Batched simulation on the GPU with MuJoCo Warp, driven entirely from torch tensors (no host round trips).

Same state convention as ``sim.BatchSim.obs_state``: p(3), quat wxyz(4), v_world(3), w_world(3), q(nj), qd(nj).
The CPU ``BatchSim`` is still used to load the model and to build the fallen-state bank (once, ~2 s).
qpos/qvel/ctrl are zero-copy torch views of the warp arrays, so acting, stepping, recording and resetting are all
asynchronous launches on the current torch stream, and the substep loop is replayed as one warp CUDA graph.
"""

from __future__ import annotations

import mujoco
import mujoco_warp as mjw
import torch
import warp as wp

from .rotations import quat_to_mat
from .sim import BatchSim


class WarpSim:
    def __init__(self, cpu: BatchSim, num_envs: int, device: str, stand_reset_frac: float = 0.0,
                 nconmax: int = 48, njmax: int = 256, sim_dt: float | None = None):  # fmt: skip
        self.m, self.N, self.nj, self.substeps = cpu.m, num_envs, cpu.nj, cpu.substeps
        self.dev, self.stand_reset_frac = device, stand_reset_frac
        wp.init()
        # optional larger timestep for training only; evaluation keeps the robot config
        self.m = m = mujoco.MjModel.from_xml_path(cpu.cfg.xml)
        m.opt.timestep, m.opt.iterations = sim_dt or cpu.m.opt.timestep, cpu.m.opt.iterations
        self.substeps = int(round(cpu.cfg.control_dt / m.opt.timestep))
        self.mw = mjw.put_model(self.m)
        self.mw.opt.graph_conditional = False  # conditional graph nodes need driver >= 12.4
        self.mw.opt.warn_overflow = 0  # solver/linesearch iteration-limit warnings fire every step; they are expected
        d = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, d, self.m.key(cpu.cfg.stand_key).id)
        mujoco.mj_forward(self.m, d)
        self.dw = mjw.put_data(self.m, d, nworld=num_envs, nconmax=nconmax, njmax=njmax)
        self.qpos, self.qvel = wp.to_torch(self.dw.qpos), wp.to_torch(self.dw.qvel)
        self.ctrl, self.warm = wp.to_torch(self.dw.ctrl), wp.to_torch(self.dw.qacc_warmstart)
        T = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)  # noqa: E731
        bank_qpos, _ = cpu.split(cpu.fallen_bank)
        self.bank = T(bank_qpos)
        self.stand_qpos, self.q_lo, self.q_hi = T(cpu.stand_qpos), T(cpu.q_lo), T(cpu.q_hi)
        self.ctrl_lo, self.ctrl_hi = T(cpu.ctrl_lo), T(cpu.ctrl_hi)
        self.stand_q = cpu.stand_q
        self._graph, self._nstep = None, 0
        self.reset(torch.ones(num_envs, dtype=torch.bool, device=device))

    def obs_state(self) -> torch.Tensor:
        qp, qv = self.qpos, self.qvel
        w_world = (quat_to_mat(qp[:, 3:7]) * qv[:, None, 3:6]).sum(-1)  # free-joint angular velocity is body-frame
        return torch.cat((qp[:, :7], qv[:, :3], w_world, qp[:, 7:], qv[:, 6:]), -1)

    def step(self, targets: torch.Tensor):
        """Hold joint position targets (N, nj) for one control period. Runs on warp's own stream (capture there works
        with warp's mempool), ordered after/before the caller's torch stream with events."""
        self.ctrl.copy_(torch.minimum(torch.maximum(targets, self.ctrl_lo), self.ctrl_hi))
        caller, ws = wp.stream_from_torch(torch.cuda.current_stream()), wp.get_stream()
        ws.wait_stream(caller)
        if self._graph is None:
            self._nstep += 1
            if self._nstep == 1:  # first call runs eagerly (loads/compiles kernels), the second captures
                for _ in range(self.substeps):
                    mjw.step(self.mw, self.dw)
            else:
                with wp.ScopedCapture() as cap:
                    for _ in range(self.substeps):
                        mjw.step(self.mw, self.dw)
                self._graph = cap.graph
        if self._graph is not None:
            wp.capture_launch(self._graph)
        caller.wait_stream(ws)

    @torch.no_grad()
    def reset(self, mask: torch.Tensor, generator: torch.Generator | None = None):
        """Branch-free reset of the envs in ``mask`` (graph-capturable): a random fallen state with small joint noise
        and zero velocity, or with probability stand_reset_frac the standing pose at a random yaw."""
        N, nj, dev = self.N, self.nj, self.dev
        rand = lambda *shape: torch.rand(*shape, device=dev, generator=generator)  # noqa: E731
        j = (rand(N) * len(self.bank)).long().clamp(max=len(self.bank) - 1)
        qp = self.bank[j]
        q = torch.minimum(torch.maximum(qp[:, 7:] + (rand(N, nj) * 0.1 - 0.05), self.q_lo), self.q_hi)
        fallen = torch.cat((qp[:, :7], q), -1)
        yaw = (rand(N) * 2 - 1) * torch.pi
        z = torch.zeros_like(yaw)
        quat = torch.stack((torch.cos(yaw / 2), z, z, torch.sin(yaw / 2)), -1)
        stand_j = self.stand_qpos[7:] + (rand(N, nj) * 0.2 - 0.1)
        stand = torch.cat((self.stand_qpos[:3].expand(N, 3), quat, stand_j), -1)
        new = torch.where((rand(N) < self.stand_reset_frac)[:, None], stand, fallen)
        m = mask[:, None]
        self.qpos.copy_(torch.where(m, new, self.qpos))
        self.qvel.copy_(torch.where(m, torch.zeros_like(self.qvel), self.qvel))
        self.warm.copy_(torch.where(m, torch.zeros_like(self.warm), self.warm))

    def contact_load(self) -> float:
        """Fraction of the shared contact buffer in use (host sync; for occasional logging only)."""
        return float(self.dw.nacon.numpy()[0]) / self.dw.naconmax
