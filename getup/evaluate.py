"""Evaluate a get-up checkpoint in the real simulator (no noise) and optionally render an mp4 grid.

python getup/evaluate.py --ckpt logs/g1_getup/last.pt --n 256 --video out.mp4
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")  # headless GPU server rendering

import mujoco
import numpy as np
import torch

from getup.mdp import GetupMDP, unflatten_state
from getup.networks import Policy
from getup.robots import ROBOTS
from getup.sim import BatchSim

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--n", type=int, default=256)
p.add_argument("--seconds", type=float, default=6.0)
p.add_argument("--seed", type=int, default=123)
p.add_argument("--video", default=None)
p.add_argument("--video_envs", type=int, default=6)
p.add_argument("--nthread", type=int, default=None)
args = p.parse_args()

ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
targs = ck["args"]
cfg = ROBOTS[targs["robot"]]
torch.set_num_threads(4)
sim = BatchSim(cfg, args.n, nthread=args.nthread, seed=args.seed)
sim.build_fallen_bank(args.n)
sim.reset(np.arange(args.n))
nj = sim.nj
T = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)  # noqa: E731
mdp = GetupMDP(T(sim.stand_q), cfg.stand_height, T(sim.q_lo), T(sim.q_hi), cfg.action_scale, cfg.control_dt)
pi = Policy(mdp.obs_dim, nj, targs["hist"])
pi.load_state_dict(ck["pi"])
pi.eval()

steps = int(args.seconds / cfg.control_dt)
a_prev = torch.zeros(args.n, nj)
traj, heights, ups = [], [], []
obs_hist = None
with torch.no_grad():
    for t in range(steps):
        S = unflatten_state(T(sim.obs_state()), nj)
        o = mdp.obs(S, a_prev)
        obs_hist = o.unsqueeze(1).repeat(1, targs["hist"], 1) if obs_hist is None else torch.cat((obs_hist[:, 1:], o.unsqueeze(1)), 1)
        a = mdp.clip_action(pi(obs_hist))
        sim.step(mdp.target(a).numpy().astype(np.float64))
        a_prev = a
        qpos, _ = sim.split(sim.states)
        traj.append(qpos[: args.video_envs].copy())
        S1 = unflatten_state(T(sim.obs_state()), nj)
        heights.append(S1[0][:, 2].numpy()), ups.append(S1[1][:, 2, 2].numpy())

H, U = np.array(heights), np.array(ups)  # (T, N)
standing = (H > 0.85 * cfg.stand_height) & (U > 0.9)
final_ok = standing[-int(1.0 / cfg.control_dt) :].all(0)  # standing for the whole last second
first = np.where(standing.any(0), standing.argmax(0) * cfg.control_dt, np.nan)
print(f"ckpt it {ck['iteration']} | success (standing through last 1 s): {final_ok.mean():.3f} | "
      f"ever stood: {standing.any(0).mean():.3f} | median time to stand {np.nanmedian(first):.2f} s | "
      f"final height {H[-1].mean():.3f} m, up {U[-1].mean():.3f}")  # fmt: skip

if args.video:
    import imageio

    m = mujoco.MjModel.from_xml_path(cfg.xml)
    d = mujoco.MjData(m)
    W, Hpx = 480, 360
    r = mujoco.Renderer(m, Hpx, W)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 3.0, -15, 120
    cols = 3
    frames = []
    for t in range(0, steps, 2):  # 25 fps
        tiles = []
        for e in range(args.video_envs):
            d.qpos[:] = traj[t][e]
            mujoco.mj_forward(m, d)
            cam.lookat[:] = (d.qpos[0], d.qpos[1], 0.5)
            r.update_scene(d, cam)
            tiles.append(r.render())
        rows = [np.concatenate(tiles[i : i + cols], 1) for i in range(0, len(tiles), cols)]
        frames.append(np.concatenate(rows, 0))
    imageio.mimsave(args.video, frames, fps=25, quality=7)
    print("wrote", args.video)

if os.environ.get("GETUP_DIAG"):
    for t in range(0, steps, 25):
        print(f"t={t * cfg.control_dt:4.1f}s standing {standing[t].mean():.2f} h {H[t].mean():.2f} up {U[t].mean():.2f}")
