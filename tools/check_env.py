"""Environment check for a new machine: imports, devices, model loading, a short batched sim benchmark,
one forward/backward of the world model, and headless rendering.

python tools/check_env.py            # ~1 min
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch

from getup.robots import ASSETS, ROBOTS
from getup.sim import BatchSim

print(f"python {sys.version.split()[0]} | mujoco {mujoco.__version__} | torch {torch.__version__} | numpy {np.__version__}")
if torch.cuda.is_available():
    print(f"cuda: {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)} | bf16 {torch.cuda.is_bf16_supported()}")
    dev = "cuda"
elif torch.backends.mps.is_available():
    print("mps available")
    dev = "mps"
else:
    print("WARNING: no GPU, networks will run on CPU")
    dev = "cpu"

# every robot config loads and its standing keyframe exists
for name, cfg in ROBOTS.items():
    m = mujoco.MjModel.from_xml_path(cfg.xml)
    m.key(cfg.stand_key)
    print(f"model {name}: nq {m.nq} nu {m.nu} OK")
for extra in ("mj_humanoid.xml", "ms_human_700/scene.xml", "ms_human_700/scene_locomotion.xml"):
    m = mujoco.MjModel.from_xml_path(os.path.join(ASSETS, extra))
    print(f"asset {extra}: nq {m.nq} nu {m.nu} OK")

# batched sim throughput
sim = BatchSim(ROBOTS["g1"], 1024)
sim.build_fallen_bank(1024)
sim.reset(np.arange(1024))
ctrl = np.tile(sim.stand_q, (1024, 1))
sim.step(ctrl)
t = time.time()
for _ in range(10):
    sim.step(ctrl)
print(f"g1 sim: {10 * 1024 / (time.time() - t):,.0f} control steps/s with {sim.nthread} threads (Mac M5 10 threads: ~46k)")

# world model fwd/bwd on the device
from getup.networks import WorldModel  # noqa: E402

def sync():
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


wm = WorldModel(29).to(dev)
x = torch.randn(1024, wm.in_dim, device=dev)
for i in range(6):
    if i == 1:
        sync()
        t = time.time()
    wm.net(x).square().mean().backward()
sync()
print(f"world model fwd+bwd (1024 rows): {(time.time() - t) / 5 * 1e3:.1f} ms on {dev} (Mac M5 MPS: ~6 ms)")

# headless render
r = mujoco.Renderer(sim.m, 120, 160)
d = mujoco.MjData(sim.m)
mujoco.mj_forward(sim.m, d)
r.update_scene(d)
img = r.render()
print(f"render OK {img.shape} (MUJOCO_GL={os.environ.get('MUJOCO_GL', 'default')})")
print("ALL CHECKS PASSED")
