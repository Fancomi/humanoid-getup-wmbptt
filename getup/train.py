"""Get-up training with the SuperMimic recipe: collect -> fit world model -> BPTT policy through the model + critic.

No reference motion: the reward is analytic in the state (uprightness, root height, standing pose), so the
same script works for any MuJoCo robot in ``robots.py``. Simulation on CPU threads (or the GPU with ``--sim warp``),
networks on MPS/CUDA. On CUDA every update phase is replayed as one CUDA graph; see train_g1_a800.sh for the fast recipe.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

T_PROC = time.time()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

import getup.networks as networks
from getup.buffer import ReplayBuffer
from getup.mdp import GetupMDP, RewardCfg, unflatten_state
from getup.networks import Critic, Policy, WorldModel
from getup.robots import ROBOTS
from getup.sim import BatchSim

p = argparse.ArgumentParser()
p.add_argument("--robot", default="g1")
p.add_argument("--num_envs", type=int, default=1024)
p.add_argument("--iterations", type=int, default=6000)
p.add_argument("--buffer_steps", type=int, default=400)
p.add_argument("--steps_per_it", type=int, default=2, help="vectorized sim steps collected per iteration")
p.add_argument("--episode_s", type=float, default=6.0)
p.add_argument("--stand_reset_frac", type=float, default=0.15, help="fraction of resets that start standing")
p.add_argument("--horizon", type=int, default=32)
p.add_argument("--hist", type=int, default=5)
p.add_argument("--wm_window", type=int, default=8)
p.add_argument("--wm_batches", type=int, default=4)
p.add_argument("--wm_batch", type=int, default=1024)
p.add_argument("--wm_hidden", type=int, default=1024)
p.add_argument("--pi_batch", type=int, default=1024)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--gamma", type=float, default=0.99)
p.add_argument("--lam", type=float, default=0.95)
p.add_argument("--action_noise", type=float, default=0.2)
p.add_argument("--critic_epochs", type=int, default=3)
p.add_argument("--critic_minibatches", type=int, default=4)
p.add_argument("--warmup_steps", type=int, default=32)
p.add_argument("--wm_only_iters", type=int, default=300)
p.add_argument("--log_every", type=int, default=25)
p.add_argument("--ckpt_every", type=int, default=500)
p.add_argument("--log_dir", default=None)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--amp", default="bf16", choices=["none", "bf16"])
p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
p.add_argument("--nthread", type=int, default=None)
p.add_argument("--bank", type=int, default=2048, help="number of fallen states in the reset bank (at least num_envs)")
p.add_argument("--sim", default="cpu", choices=["cpu", "warp"], help="warp = MuJoCo Warp on the GPU (cuda only)")
p.add_argument("--sim_dt", type=float, default=None, help="physics timestep for training (evaluation keeps the robot config)")
p.add_argument("--overlap", type=int, default=None, help="collect on a second CUDA stream while the update graphs run "
               "(default: on for cpu sim with --graph; off for warp, whose single-stream loop never blocks the host)")
p.add_argument("--reward", default="", help="RewardCfg overrides, e.g. w_up=1.5,w_height=4")
p.add_argument("--compile", type=int, default=None, help="torch.compile the per-step loop bodies (default: on for cuda)")
p.add_argument("--graph", type=int, default=None, help="capture each whole update phase in one CUDA graph (default: on for cuda)")
args = p.parse_args()
if args.compile is None:
    args.compile = int(args.device.startswith("cuda"))
if args.graph is None:
    args.graph = int(args.device.startswith("cuda"))
if args.overlap is None:
    args.overlap = int(bool(args.graph) and args.sim == "cpu")
p_overlap = args.overlap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
args.log_dir = args.log_dir or os.path.join(REPO, "logs", f"{args.robot}_getup")
os.makedirs(args.log_dir, exist_ok=True)
json.dump(vars(args), open(os.path.join(args.log_dir, "args.json"), "w"), indent=2)
torch.manual_seed(args.seed)
networks.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
dev = args.device
cfg = ROBOTS[args.robot]
if args.sim == "cpu" and args.sim_dt:
    cfg = dataclasses.replace(cfg, sim_dt=args.sim_dt)

# ------------------------------------------------------------------------------------------- setup
sim = BatchSim(cfg, args.num_envs, nthread=args.nthread, seed=args.seed)
t0 = time.time()
bank = sim.build_fallen_bank(max(args.bank, args.num_envs))
print(f"fallen bank: {len(bank)} states in {time.time() - t0:.1f}s, root height {sim.split(bank)[0][:, 2].mean():.3f} m", flush=True)
nj = sim.nj
T = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)  # noqa: E731
rew_cfg = RewardCfg(**{k: float(v) for k, v in (kv.split("=") for kv in args.reward.split(",") if kv)})
mdp = GetupMDP(T(sim.stand_q), cfg.stand_height, T(sim.q_lo), T(sim.q_hi), cfg.action_scale, cfg.control_dt, rew_cfg)
STATE_DIM, OBS_DIM, HZ, Hh = 13 + 2 * nj, mdp.obs_dim, args.horizon, args.hist
ep_len = int(args.episode_s / cfg.control_dt)

wm = WorldModel(nj, hidden=(args.wm_hidden,) * 3).to(dev)
pi = Policy(OBS_DIM, nj, Hh).to(dev)
critic = Critic(mdp.critic_dim()).to(dev)
if p_overlap:
    # collection gets its own policy and normalizer copies, synced once per iteration, so the collection stream never
    # reads a tensor the update graphs are writing
    pi_act = copy.deepcopy(pi).requires_grad_(False)  # acting policy; its obs_norm is the master copy
    in_norm_c, out_norm_c = copy.deepcopy(wm.in_norm), copy.deepcopy(wm.out_norm)
else:
    pi_act, in_norm_c, out_norm_c = pi, wm.in_norm, wm.out_norm
fused = dev.startswith("cuda")
# graph capture needs the lr and the step count on the device
LR = (lambda: torch.tensor(args.lr, device=dev)) if args.graph else (lambda: args.lr)  # noqa: E731
opt_kw = dict(fused=fused, capturable=bool(args.graph))
opt_wm = torch.optim.Adam(wm.parameters(), lr=LR(), **opt_kw)
opt_pi = torch.optim.Adam(pi.parameters(), lr=LR(), betas=(0.9, 0.95), **opt_kw)
opt_v = torch.optim.Adam(critic.parameters(), lr=LR(), **opt_kw)
# fused Adam skips the step where found_inf is 1, without a host sync on the grad norm
opt_pi.found_inf = torch.zeros((), device=dev) if fused else None
buffer = ReplayBuffer(args.buffer_steps, args.num_envs, STATE_DIM, nj, dev)

ERR_FLOOR = torch.cat((torch.full((3,), 3e-3), torch.full((3,), 5e-3), torch.full((3,), 2e-2), torch.full((3,), 5e-2),
                       torch.full((nj,), 3e-3), torch.full((nj,), 5e-2))).to(dev)  # fmt: skip
US = lambda x: unflatten_state(x, nj)  # noqa: E731


def _wm_step_err(S, a, s1):
    S = wm.step(S, a)
    return S, wm.state_error(US(s1), S)


def _imag_step(S, o_hist, a_p):
    a = mdp.clip_action(pi(o_hist))
    S = wm.step(S, a)
    r = mdp.reward(S, a, a_p)
    o_hist = torch.cat((o_hist[:, 1:], mdp.add_noise(mdp.obs(S, a)).unsqueeze(1)), 1)
    return S, a, r, o_hist


def _wm_loss(s, a, s1, mask):
    """Whole k-step world-model rollout loss; with --graph this is compiled as one graph (fewer, fused kernels)."""
    S = US(s[:, 0])
    errs = []
    for j in range(args.wm_window):
        S = wm.step(S, a[:, j])
        errs.append(wm.state_error(US(s1[:, j]), S))
    std = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
    e = ((torch.stack(errs, 1) / std) ** 2).mean(-1)  # (B, k)
    return (e * mask).sum() / args.wm_batch, e[:, -1].detach()


def _critic_loss(x, y):
    return ((critic(x) - y) ** 2).mean()


# reduce-overhead = per-step CUDA graphs (tiny graphs: launch overhead dominates without them). With --graph the
# whole update phase is captured by hand instead, so compile only fuses kernels.
cmode = "default" if args.graph else "reduce-overhead"
maybe_compile = (lambda f: torch.compile(f, mode=cmode)) if args.compile else (lambda f: f)
wm_step_err, imag_step, critic_loss = maybe_compile(_wm_step_err), maybe_compile(_imag_step), maybe_compile(_critic_loss)
wm_loss = maybe_compile(_wm_loss) if args.graph else None


def lr_at(it):
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(it / args.iterations, 1.0))))


def set_requires_grad(m, flag):
    for q in m.parameters():
        q.requires_grad_(flag)


def set_lr(opt, v):
    for g in opt.param_groups:
        if torch.is_tensor(g["lr"]):
            g["lr"].fill_(v)
        else:
            g["lr"] = v


def cuda_graphed(fn, warmup=3):
    """Replay ``fn`` (a full update: sample, forward, backward, optimizer step) as one CUDA graph. The first call
    runs ``warmup`` real updates eagerly on a side stream (allocates optimizer state and grads), then captures.
    Grads are zeroed in place inside ``fn``, so they keep their addresses across replays."""
    state = {}

    def call():
        if "g" not in state:
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(warmup):
                    fn()
            torch.cuda.current_stream().wait_stream(st)
            state["g"] = torch.cuda.CUDAGraph()
            # own capture stream per graph: the default capture stream is shared, and cuBLAS keys its workspace by
            # stream, so graphs captured on it would share one workspace and corrupt each other when run concurrently
            with torch.cuda.graph(state["g"], stream=torch.cuda.Stream()):
                state["out"] = fn()
        state["g"].replay()
        return tuple(o.clone() for o in state["out"])

    return call if args.graph else fn


# ------------------------------------------------------------------------------------------- env loop
N = args.num_envs
SD = STATE_DIM
rng = np.random.default_rng(args.seed + 1)
ep_step = np.zeros(N, dtype=np.int64)  # episode bookkeeping on the host
first_act = True


def reset_envs(ids: np.ndarray):
    sim.reset(ids)
    n_st = int(round(args.stand_reset_frac * len(ids)))
    if n_st:
        st_ids = rng.choice(ids, n_st, replace=False)
        qpos = np.tile(sim.stand_qpos, (n_st, 1))
        yaw = rng.uniform(-np.pi, np.pi, n_st)
        qpos[:, 3], qpos[:, 6] = np.cos(yaw / 2), np.sin(yaw / 2)
        qpos[:, 7:] += rng.uniform(-0.1, 0.1, (n_st, nj))
        sim.states[st_ids] = sim._full_from_qpos_qvel(qpos, np.zeros((n_st, sim.m.nv)))


reset_envs(np.arange(N))
sim_steps = 0
sim_pool = ThreadPoolExecutor(1)

# One packed copy each way per sim step. Host side: pinned staging; device side: persistent tensors that the
# (graph-captured) acting and recording functions read and write in place.
pin = {"pin_memory": True} if dev.startswith("cuda") else {}
h_act, h_rec, h_tgt = torch.zeros(N, SD + 1, **pin), torch.zeros(N, SD + 3, **pin), torch.zeros(N, nj, **pin)
d_act, d_rec = torch.zeros(N, SD + 1, device=dev), torch.zeros(N, SD + 3, device=dev)  # [s, fresh], [s1, bad, done, ep_step]
a_prev, a_cur, tgt = (torch.zeros(N, nj, device=dev) for _ in range(3))
obs_hist = torch.zeros(N, Hh, OBS_DIM, device=dev)
acc = torch.zeros(6, device=dev)  # sum of: mean reward, mean height, #bad, #standing at episode end, #ended, #steps
# transitions are staged on the device; overlapped mode commits them to the replay buffer at the next iteration
staging = {k: torch.zeros((args.steps_per_it,) + v.shape[1:], dtype=v.dtype, device=dev) for k, v in buffer.data.items()}
slot = torch.zeros(1, dtype=torch.long, device=dev)
staging_on, n_staged = False, 0


def commit_staged():
    global n_staged
    for i in range(n_staged):
        buffer.add({k: v[i] for k, v in staging.items()})
    n_staged = 0
    slot.zero_()


# Collection draws from its own CUDA generator, so the acting graph on the collection stream and the update graphs on
# the main stream never advance the same generator state concurrently.
gen_col = torch.Generator(device=dev) if dev.startswith("cuda") else None
if gen_col is not None:
    gen_col.manual_seed(args.seed + 12345)


@torch.no_grad()
def _act():
    s, fresh = d_act[:, :SD], d_act[:, SD] > 0.5
    o = mdp.add_noise(mdp.obs(US(s), a_prev), generator=gen_col)
    obs_hist.copy_(torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(obs_hist), torch.cat((obs_hist[:, 1:], o.unsqueeze(1)), 1)))
    pi_act.obs_norm.update(o)
    a_cur.copy_(mdp.clip_action(pi_act(obs_hist) + args.action_noise * torch.randn(N, nj, device=dev, generator=gen_col)))
    tgt.copy_(mdp.target(a_cur))


@torch.no_grad()
def _record():
    s, s1 = d_act[:, :SD], d_rec[:, :SD]
    bad, done, ep = d_rec[:, SD] > 0.5, d_rec[:, SD + 1] > 0.5, d_rec[:, SD + 2].long()
    S, S1 = US(s), US(s1)
    for k, v in (("s", s), ("s1", s1), ("a", a_cur), ("a_prev", a_prev), ("ep_step", ep), ("done", done)):
        staging[k].index_copy_(0, slot, v.unsqueeze(0))
    slot.add_(1)
    in_norm_c.update(wm.inputs(S, a_cur))
    out_norm_c.update(wm.targets(S, S1), w=~bad)
    r, terms = mdp.reward(S1, a_cur, a_prev, terms=True)
    ended = (done & ~bad).float()
    acc.add_(torch.stack((r.mean() / mdp.max_step_reward, terms["height"].mean(), bad.float().sum(),
                          (terms["standing"].gt(0.5).float() * ended).sum(), ended.sum(), torch.ones((), device=dev))))  # fmt: skip
    a_prev.copy_(torch.where(done[:, None], torch.zeros_like(a_cur), a_cur))


def graph_once(fn):
    """First call runs eagerly (a real step that also warms up allocations); the second captures ``fn`` into a CUDA
    graph, and every call from then on replays it. ``fn`` must only read/write persistent tensors in place."""
    state = {"n": 0}

    def call():
        if "g" not in state:
            state["n"] += 1
            if state["n"] == 1:
                return fn()
            state["g"] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(state["g"], stream=torch.cuda.Stream()):
                fn()
        state["g"].replay()

    return call if args.graph else fn


run_act, run_record = graph_once(_act), graph_once(_record)


def to_host(dst, src):
    dst.copy_(src, non_blocking=True)
    if dev.startswith("cuda"):
        torch.cuda.current_stream().synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


def collect_begin():
    """Pick actions on the device and start the CPU sim step in a background thread (mujoco.rollout releases the
    GIL), so it runs while the GPU updates. Only the sim thread touches ``sim`` until ``collect_end``."""
    global first_act
    s_np = sim.obs_state()
    h_act[:, :SD] = torch.from_numpy(s_np)
    h_act[:, SD] = torch.from_numpy(((ep_step == 0) | first_act).astype(np.float32))
    first_act = False
    d_act.copy_(h_act, non_blocking=True)
    run_act()
    to_host(h_tgt, tgt)
    return s_np, sim_pool.submit(sim.step, h_tgt.numpy().astype(np.float64))


def collect_end(pending):
    global ep_step, sim_steps, n_staged
    s_np, fut = pending
    fut.result()
    s1_np = sim.obs_state()
    with np.errstate(invalid="ignore"):
        bad = ~np.isfinite(s1_np).all(-1) | (np.abs(s1_np[:, 7:13]) > 50).any(-1)
    s1_np[bad] = s_np[bad]
    done = (ep_step + 1 >= ep_len) | bad
    h_rec[:, :SD] = torch.from_numpy(s1_np)
    h_rec[:, SD], h_rec[:, SD + 1], h_rec[:, SD + 2] = (torch.from_numpy(x.astype(np.float32)) for x in (bad, done, ep_step))
    d_rec.copy_(h_rec, non_blocking=True)
    run_record()
    n_staged += 1
    if not staging_on:
        commit_staged()
    ep_step = np.where(done, 0, ep_step + 1)
    ids = np.nonzero(done)[0]
    if len(ids):
        reset_envs(ids)
    sim_steps += N


def collect(n_steps):
    if wsim is not None:
        return collect_warp(n_steps)
    for _ in range(n_steps):
        collect_end(collect_begin())


# ---- GPU simulation: the whole step (observe, act, simulate, record, reset) is device-side and asynchronous
wsim = None
if args.sim == "warp":
    from getup.sim_warp import WarpSim

    wsim = WarpSim(sim, N, dev, stand_reset_frac=args.stand_reset_frac, sim_dt=args.sim_dt)
    ep_d = torch.as_tensor(ep_step, device=dev)
    fresh_d = torch.ones(N, dtype=torch.bool, device=dev)


@torch.no_grad()
def _pre_warp():
    d_act[:, :SD].copy_(wsim.obs_state())
    d_act[:, SD].copy_(fresh_d.float())
    _act()


@torch.no_grad()
def _post_warp():
    s, s1 = d_act[:, :SD], wsim.obs_state()
    bad = ~torch.isfinite(s1).all(-1) | (s1[:, 7:13].abs() > 50).any(-1)
    done = (ep_d + 1 >= ep_len) | bad
    d_rec[:, :SD].copy_(torch.where(bad[:, None], s, s1))
    d_rec[:, SD].copy_(bad.float()), d_rec[:, SD + 1].copy_(done.float()), d_rec[:, SD + 2].copy_(ep_d.float())
    _record()
    ep_d.copy_(torch.where(done, torch.zeros_like(ep_d), ep_d + 1))
    fresh_d.copy_(done)
    wsim.reset(done, generator=gen_col)


run_pre_warp, run_post_warp = graph_once(_pre_warp), graph_once(_post_warp)


def collect_warp(n_steps):
    global sim_steps, n_staged
    for _ in range(n_steps):
        run_pre_warp()
        wsim.step(tgt)
        run_post_warp()
        n_staged += 1
        if not staging_on:
            commit_staged()
        sim_steps += N


# ------------------------------------------------------------------------------------------- updates
def wm_update():
    torch.compiler.cudagraph_mark_step_begin()
    w = buffer.sample_windows(args.wm_batch, args.wm_window)
    if wm_loss is not None:
        loss, e_last = wm_loss(w["s"], w["a"], w["s1"], w["mask"])
    else:
        S = US(w["s"][:, 0])
        errs = []
        for j in range(args.wm_window):
            S, err = wm_step_err(S, w["a"][:, j], w["s1"][:, j])
            errs.append(err)
        std = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
        e = ((torch.stack(errs, 1) / std) ** 2).mean(-1)  # (B, k)
        loss, e_last = (e * w["mask"]).sum() / args.wm_batch, e[:, -1].detach()
    opt_wm.zero_grad(set_to_none=not args.graph)
    loss.backward()
    nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
    opt_wm.step()
    m = w["mask"][:, -1]
    return loss.detach() / args.wm_window, (e_last * m).sum() / m.sum().clamp(min=1)


def policy_update():
    torch.compiler.cudagraph_mark_step_begin()
    s, h = buffer.sample_starts(args.pi_batch, Hh)
    S, a_p = US(s["s"]), s["a_prev"]
    with torch.no_grad():
        o_hist = mdp.add_noise(mdp.obs(US(h["s"]), h["a_prev"]))
    set_requires_grad(wm, False), set_requires_grad(critic, False)
    B = args.pi_batch
    S0, Ss, rs = S, [], []
    for t in range(HZ):
        S, a, r_t, o_hist = imag_step(S, o_hist, a_p)
        rs.append(r_t)
        a_p = a
        Ss.append(S)
    r = torch.stack(rs)  # (H, B)
    disc = args.gamma ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
    J = (disc * r).sum(0) + args.gamma**HZ * critic(mdp.critic_features(S))
    loss = -J.mean()
    opt_pi.zero_grad(set_to_none=not args.graph)
    loss.backward()
    gn = nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
    if opt_pi.found_inf is not None:
        opt_pi.found_inf.copy_((~torch.isfinite(gn)).float())
        opt_pi.step()
    elif torch.isfinite(gn):
        opt_pi.step()
    set_requires_grad(wm, True), set_requires_grad(critic, True)
    # critic: TD(lambda) targets on the imagined trajectory
    with torch.no_grad():
        S_c = tuple(torch.cat((S0[i].unsqueeze(0), torch.stack([St[i].detach() for St in Ss])), 0).flatten(0, 1) for i in range(6))
        X = mdp.critic_features(S_c).view(HZ + 1, B, -1)
        critic.norm.update(X)
        V = critic(X)
        R, rews, targets = V[-1], r.detach(), []
        for t in range(HZ - 1, -1, -1):
            R = rews[t] + args.gamma * ((1 - args.lam) * V[t + 1] + args.lam * R)
            targets.append(R)
        targets = torch.stack(targets[::-1])
    Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
    n, mb = Xf.shape[0], Xf.shape[0] // args.critic_minibatches
    for _ in range(args.critic_epochs):
        perm = torch.argsort(torch.rand(n, device=dev))  # randperm is not graph-capturable
        for i in range(args.critic_minibatches):
            idx = perm[i * mb : (i + 1) * mb]
            torch.compiler.cudagraph_mark_step_begin()
            v_loss = critic_loss(Xf[idx], Yf[idx])
            opt_v.zero_grad(set_to_none=not args.graph)
            v_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opt_v.step()
    return loss.detach(), v_loss.detach(), (r.detach().mean() / mdp.max_step_reward), gn.detach()


def wm_phase():
    for _ in range(args.wm_batches):
        out = wm_update()
    return out


run_wm_phase, run_pi_phase = cuda_graphed(wm_phase), cuda_graphed(policy_update)
run_wm_one = cuda_graphed(wm_update)  # one world-model update per graph, for fine-grained interleaving with the sim


def save(it, wall):
    ck = {"wm": wm.state_dict(), "pi": pi.state_dict(), "critic": critic.state_dict(), "iteration": it, "wall_time": wall,
          "sim_steps": sim_steps, "args": vars(args)}  # fmt: skip
    torch.save(ck, os.path.join(args.log_dir, "last.pt"))
    torch.save({k: ck[k] for k in ("pi", "iteration", "wall_time", "sim_steps", "args")}, os.path.join(args.log_dir, f"ckpt_{it:06d}.pt"))


def interleaved_iteration(it):
    """Single stream, CPU sim: while sim step k runs on the CPU, the GPU works through a slice of this iteration's
    updates (world-model updates spread over the first steps, the policy update during the last), so the next action
    only waits behind a slice about as long as one sim step and both CPU and GPU stay busy."""
    S_ = args.steps_per_it
    pi_on = it >= args.wm_only_iters
    n_wm_steps = S_ - 1 if (pi_on and S_ > 1) else S_
    per = [args.wm_batches // n_wm_steps + (1 if k < args.wm_batches % n_wm_steps else 0) for k in range(n_wm_steps)]
    for k in range(S_):
        pending = collect_begin()
        if k < n_wm_steps:
            for _ in range(per[k]):
                l, e8 = run_wm_one()
        if pi_on and k == S_ - 1:
            pl, vl, ir, gn = run_pi_phase()
            agg["pi"].append(pl), agg["v"].append(vl), agg["ir"].append(ir), agg["gn"].append(gn)
        collect_end(pending)
    agg["wm"].append(l), agg["wm8"].append(e8)


def sync_iteration(it):
    """Single-stream loop. Without CUDA graphs (MPS/CPU): the first sim step runs during the world-model updates, the
    second during the policy update."""
    if wsim is None and args.graph:
        return interleaved_iteration(it)
    if wsim is not None:
        collect(args.steps_per_it)
        l, e8 = run_wm_phase()
        agg["wm"].append(l), agg["wm8"].append(e8)
        if it >= args.wm_only_iters:
            pl, vl, ir, gn = run_pi_phase()
            agg["pi"].append(pl), agg["v"].append(vl), agg["ir"].append(ir), agg["gn"].append(gn)
        return
    pending = collect_begin()
    l, e8 = run_wm_phase()
    agg["wm"].append(l), agg["wm8"].append(e8)
    collect_end(pending)
    for k in range(args.steps_per_it - 1):
        pending = collect_begin()
        if k == 0 and it >= args.wm_only_iters:
            pl, vl, ir, gn = run_pi_phase()
            agg["pi"].append(pl), agg["v"].append(vl), agg["ir"].append(ir), agg["gn"].append(gn)
        collect_end(pending)
    if args.steps_per_it == 1 and it >= args.wm_only_iters:
        pl, vl, ir, gn = run_pi_phase()
        agg["pi"].append(pl), agg["v"].append(vl), agg["ir"].append(ir), agg["gn"].append(gn)


if p_overlap:
    col_stream = torch.cuda.Stream(priority=-1)
    sync_src = [*pi_act.obs_norm.buffers(), *in_norm_c.buffers(), *out_norm_c.buffers(), *pi.parameters()]
    sync_dst = [*pi.obs_norm.buffers(), *wm.in_norm.buffers(), *wm.out_norm.buffers(), *pi_act.parameters()]


def overlap_iteration(it):
    """World-model and policy graphs on the main stream, CPU sim + acting on a high-priority collection stream,
    concurrently. The host only blocks inside collection (sim + its own small syncs). The policy graph stays on the main
    stream: replaying it concurrently with the world-model graph on a third stream hit illegal memory accesses."""
    main = torch.cuda.current_stream()
    main.wait_stream(col_stream)  # previous iteration's transitions and normalizer statistics
    commit_staged()
    with torch.no_grad():
        torch._foreach_copy_(sync_dst, sync_src)
    col_stream.wait_stream(main)
    l, e8 = run_wm_phase()
    agg["wm"].append(l), agg["wm8"].append(e8)
    if it >= args.wm_only_iters:
        pl, vl, ir, gn = run_pi_phase()
        agg["pi"].append(pl), agg["v"].append(vl), agg["ir"].append(ir), agg["gn"].append(gn)
    with torch.cuda.stream(col_stream):
        collect(args.steps_per_it)


# ------------------------------------------------------------------------------------------- main
collect(args.warmup_steps)
acc.zero_()
log_f = open(os.path.join(args.log_dir, "train_log.csv"), "w")
log_f.write("iteration,wall_min,sim_steps,wm_loss,wm_err8,pi_loss,v_loss,imag_rew,gradnorm,sim_rew,height,success,it_per_s\n")
t_start = t_log = time.time()
staging_on = bool(p_overlap)
agg = {k: [] for k in ("wm", "wm8", "pi", "v", "ir", "gn")}
mean = lambda x: float(sum(float(v) for v in x) / max(len(x), 1))  # noqa: E731
for it in range(args.iterations):
    for o in (opt_wm, opt_pi, opt_v):
        set_lr(o, lr_at(it))
    overlap_iteration(it) if p_overlap else sync_iteration(it)
    if (it + 1) % args.log_every == 0:
        if p_overlap:
            col_stream.synchronize()  # acc is written on the collection stream
        r_m, h_m, n_bad, n_up, n_end, n_st = acc.tolist()
        acc.zero_()
        ips = args.log_every / (time.time() - t_log)
        t_log = time.time()
        wall = (time.time() - t_start) / 60
        row = [it + 1, wall, sim_steps, mean(agg["wm"]), mean(agg["wm8"]), mean(agg["pi"]), mean(agg["v"]), mean(agg["ir"]),
               mean(agg["gn"]), r_m / max(n_st, 1), h_m / max(n_st, 1), n_up / max(n_end, 1), ips]  # fmt: skip
        log_f.write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row) + "\n")
        log_f.flush()
        print(f"it {it + 1} | {wall:.1f} min | wm {row[3]:.4f} e8 {row[4]:.4f} | pi {row[5]:.3f} v {row[6]:.4f} imag_r {row[7]:.3f} "
              f"gn {row[8]:.2f} | sim_r {row[9]:.3f} h {row[10]:.3f} succ {row[11]:.2f} bad {int(n_bad)} "
              f"| {ips:.2f} it/s", flush=True)  # fmt: skip
        agg = {k: [] for k in agg}
    if (it + 1) % args.ckpt_every == 0 or it + 1 == args.iterations:
        save(it + 1, time.time() - t_start)
log_f.close()
print(f"total {time.time() - T_PROC:.1f} s (training loop {(time.time() - t_start) / 60:.2f} min)", flush=True)
# skip interpreter teardown (CUDA graphs, compiled kernels, sim thread pool): ~2 s, nothing left to flush
os._exit(0)
