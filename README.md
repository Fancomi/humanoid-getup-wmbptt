# humanoid-getup-wmbptt

把 [SuperMimic](https://github.com/zita-ch/supermimic_vibed)（@ ee0608f，MIT）的训练方法（在学到的世界模型里做 BPTT）用在**不跟踪参考动作的单一技能**上，第一个任务是从任意倒地姿态起身。

原仓库依赖 Isaac Lab 和 CUDA。这里把物理仿真换成 CPU 多线程的 `mujoco.rollout`，网络放在 torch 上，Mac (MPS) 和 Linux (CUDA) 都能跑。奖励只依赖状态（躯干直立、根节点高度，站起来之后再加站姿和静止两项），不需要正运动学和参考动作，所以换模型只需要在 `getup/robots.py` 里加一条配置。

## 已有结果

Mac M5（MPS，1024 个环境，6000 次迭代，默认参数）：

| 模型 | 关节数 | 训练用时 | 起身成功率* | 起身用时中位数 |
|---|---|---|---|---|
| Unitree G1 | 29 | 37.4 分钟 | 94.7% | 1.08 s |
| SMPL 人形 | 69 | 44 分钟 | 93.8% | 0.48 s |

A800 单卡（`tools/train_g1_a800.sh`，2300 次迭代）：G1 整个进程约 108 到 117 秒（单独占卡、CPU 负载 70 到 115 时实测），约为 Mac 的 1/20；成功率因运行而异，9 轮里 4 轮 ≥ 95%（最高 99.8%），其余 8% 到 93%。

\* 512 个随机倒地初始状态，`--seed 7`，标准是最后 1 秒一直保持站立（根节点高度 > 0.85 × 站立高度，且直立度 up > 0.9）。

成功率要到训练后段才明显上升，之前几乎是 0，这是正常现象。动作偏猛（踢腿、翻滚），因为奖励里没有能耗和接触冲击惩罚，也没有做域随机化，所以**不能直接用于真机**。checkpoint、日志和视频都在 `logs/`，不入库。

## 在 A800 上开始

```bash
bash setup_a800.sh                 # 建 .venv，装 CUDA 版 torch 和其他依赖，运行 tools/check_env.py
                                   # torch 2.14.1 在 download.pytorch.org 上只有 cu126 版本

GPU=1 bash tools/train_g1_a800.sh logs/g1_fast        # G1 快速配方：训练约 2 分钟，然后评估并出视频

# 默认参数（与 Mac 结果同配置，CUDA 上约 10 分钟）
.venv/bin/python getup/train.py --robot g1 --log_dir logs/g1
.venv/bin/python getup/evaluate.py --ckpt logs/g1/last.pt --n 512 --seed 7 --video logs/g1/getup.mp4
.venv/bin/python getup/train.py --robot smpl --log_dir logs/smpl
bash tools/eval_ckpts.sh logs/g1 3000                 # 评估目录里第 3000 次以后的每个 checkpoint
```

`tools/check_env.py` 会检查依赖版本、GPU、所有模型能否加载、仿真吞吐、世界模型一次前向加反向的耗时，以及无显示器渲染。最后一行打印 `ALL CHECKS PASSED` 就表示环境正常。渲染需要 EGL（Ubuntu 上是 `libegl1`），只影响出视频，不影响训练。

**训练的每一行日志**依次是：世界模型误差（`wm`，以及第 8 步的累积误差 `e8`）、策略损失 `pi`、critic 损失 `v`、模型里想象出来的奖励 `imag_r`、真实仿真里的奖励 `sim_r`、平均根节点高度 `h`、回合结束时的站立比例 `succ`、每秒迭代次数。重点看 `e8` 有没有下降、`sim_r` 有没有上升。

## CUDA 上的实现要点

- **CUDA graph**：每一步的小算子受 kernel 启动开销限制，不用图时 CUDA 上只有 1.3 it/s，比 Mac 还慢。`--graph`（CUDA 上默认开）把世界模型更新阶段和策略更新阶段各捕获成一张图；世界模型 8 步窗口用 `torch.compile` 编译成一个函数；采集里的观测、动作、记录也各是一张图。为了能捕获：Adam 用 `fused` + `capturable`，学习率是设备上的张量，经验回放采样和 `RunningNorm` 更新都不做 host 同步。
- **仿真与更新并行**：CPU 仿真在后台线程跑（`mujoco.rollout` 释放 GIL）。`--overlap 1`（CPU 仿真时默认开）把采集放在独立 stream 上，与更新图并发；采集用自己的策略副本、归一化副本和随机数生成器，每次迭代开头同步一次，新数据在下一次迭代才进经验回放。
- **快速配方**（`tools/train_g1_a800.sh`）：`--steps_per_it 3 --wm_batches 8 --wm_hidden 768 --lr 6e-4 --action_noise 0.3 --sim_dt 0.005 --bank 1024 --iterations 2300 --nthread 48 --overlap 1`。`--sim_dt 0.005` 只改训练用的物理步长，评估仍用 `robots.py` 里的 0.004。单独占卡时约 20 it/s；速度主要看机器上空闲的 CPU 核数，G1 1024 个环境用 32 到 64 线程最快，128 线程反而更慢；GPU 被别的任务占用时会明显变慢。
- **限制**：结果不确定（异步仿真和 CUDA graph 都不是确定性的，同一 seed 重跑也会不同），约一半运行能到 95%，失败的停在弓步或单腿撑地的局部最优。加大 batch 或环境数、每次迭代多次策略更新、EMA、调高 `w_up`、噪声 0.4/0.5 都试过，没有改善或更差。
- **已知问题**：`--overlap 1` 在同一张卡上同时跑多个训练进程时偶尔报 CUDA illegal memory access，根因未查明；单独跑时未出现。出现时可用 `--overlap 0`（单 stream，慢约 30%）。
- **GPU 仿真**：`--sim warp`（`getup/sim_warp.py`，MuJoCo Warp），与 CPU 仿真逐步对得上。驱动 12.2 需装 GitHub release 的 `warp_lang-1.18.0+cu12` 和 `mujoco-warp==3.17.0`（见 `requirements.txt`），并关闭 `graph_conditional`（代码里已关）。在这台机器上端到端不比 CPU 仿真快，因为网络更新和仿真抢同一张卡。

## 目录结构

```
getup/
  robots.py     每个模型一条配置：场景 xml、站立关键帧、站立高度、动作缩放、仿真步长
  sim.py        批量仿真：mujoco.rollout 多线程；生成倒地状态库（从随机朝向掉落，驱动减弱后落地稳定）；重置
  mdp.py        观测、奖励、critic 特征，都是可微分的解析式，不依赖具体模型
  networks.py   世界模型（预测局部坐标系下的状态增量）、策略、critic、RunningNorm
  buffer.py     经验回放：按 (时间, 环境) 存储，采样长度为 k 的窗口和带历史的起点
  train.py      主循环：采集 → 拟合世界模型 → 在模型里沿 32 步做 BPTT 更新策略 → 用 TD(λ) 更新 critic；结束时打印 total 用时
  sim_warp.py   可选的 GPU 批量仿真（MuJoCo Warp），接口与 sim.py 对齐，--sim warp 启用
  evaluate.py   在真实仿真里评估（不加噪声）并渲染多格 mp4；设 GETUP_DIAG=1 会打印站立比例随时间的变化
  rotations.py  旋转工具函数（改自 SuperMimic，MIT，版权声明见文件头）
tools/
  check_env.py     新机器环境自检
  train_g1_a800.sh G1 快速配方入口（训练 + 评估 + 视频）
  eval_ckpts.sh    评估一个目录里的所有 checkpoint
  smpl_to_scene.py 把 ProtoMotions 的 SMPL MJCF 转成可训练的场景：加关节限位、位置伺服、地面、站立关键帧
assets/
  unitree_g1/       Menagerie G1 的 MJX 版（mujoco_menagerie @ feadf76，BSD-3-Clause）
  smpl_humanoid_raw.xml / smpl_scene.xml   ProtoMotions 原始 SMPL 模型和转换后的场景
  mj_humanoid.xml   MuJoCo 自带的 humanoid（力矩电机，还没接进 robots.py）
  ms_human_700/     MS-Human-700 肌肉骨骼模型（mujoco_menagerie @ f3b6013，Apache-2.0，还没接入）
logs/             训练产物（checkpoint、train_log.csv、stdout、视频），不入库
```

## 尚未完成的工作和已知问题

- **MuJoCo humanoid**：已经测过用 `MjSpec` 把力矩电机改成位置伺服后仿真稳定（kp = 4 × gear，力的上限沿用原 gear，关节范围要从角度换算成弧度）。但让所有关节保持零位时，它 3 秒内会自己倒下，需要策略主动保持平衡，或者换一个站姿。在 robots.py 里加配置之前，要先写一个类似 `smpl_to_scene.py` 的转换脚本。
- **MS-Human-700**：能加载、能仿真。要接入还需要：(1) 骨盆是 3 个平移加 3 个转动关节，不是自由关节，`sim.py` 默认 `qpos[:7]` 是位置加四元数，需要转换；(2) 肌肉激活状态（`m.na` 个）要加进状态，或者在 torch 里用解析的一阶激活动力学来算；(3) 膝关节有从属自由度，由等式约束绑在膝角上，可以从状态里去掉；(4) 下肢版的上半身没有关节，没法用手撑地，做起身得用全身版。仿真速度（Mac 10 线程，1024 个环境）：下肢版约 8900 控制步/秒，全身版约 2400，分别约为 G1 的 1/5 和 1/20。
- **走路任务**：建议路线是先在 MuJoCo humanoid 上用速度目标验证走路奖励，再在肌肉下肢版上先做站立平衡，然后跟踪一段参考步态（原仓库 `mdp.py` 里的跟踪奖励可以复用），最后去掉参考，只用奖励加肌肉用力惩罚。
- `sim.py` 里判定"已倒地"的阈值已经改成 `0.6 × stand_height`，不再是 G1 专用的 0.45 m。
- 采集时如果某个环境的状态出现非有限值或速度过大（`bad`），这一步的转移会被丢弃，环境被重置。两轮正式训练里偶尔出现几次，没有影响结果。

## 版本

Python 3.12，mujoco 3.17.0，torch 2.14.1，numpy 2.5.4（见 `requirements.txt`）。checkpoint 是普通的 `torch.save` 字典，在 CPU 和 CUDA 上都能用 `map_location` 加载。
