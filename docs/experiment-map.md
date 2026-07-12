# Experiment map

This document describes the experiment families present in the repository as of the reproducibility baseline. Statements below are based on direct inspection of the source tree. Unverified items are marked explicitly.

## Directory overview

| Directory | Role |
| --- | --- |
| `MLP/` | PPO + `MlpPolicy` experiments on the CRTBP ARPOD environment |
| `LSTM/` | RecurrentPPO + LSTM policy experiments on a closely related environment |
| `MLPconstAng/` | PPO + MLP with an expanded observation and additional operational constraints |
| `LSTMconstAng/` | RecurrentPPO + LSTM with the constant-angle / expanded-constraint environment |
| `ExtraCode/` | Historical / classical baselines and plotting utilities (OCP, LQR, older LSTM) |

## Environment classes

All formal RL environments expose class `ArpodCrtbp(gym.Env)`.

| Variant key | Module path | Observation shape | Action shape | Gym API |
| --- | --- | --- | --- | --- |
| MLP nominal | `MLP/Environment.py::ArpodCrtbp` | `(16,)` | `(3,)` | `reset() -> obs`; `step() -> (obs, reward, done, info)` |
| LSTM nominal | `LSTM/Environment.py::ArpodCrtbp` | `(16,)` | `(3,)` | same |
| MLP perturbed | `MLP/EnvironmentPert.py::ArpodCrtbp` | `(16,)` | `(3,)` | same |
| LSTM perturbed | `LSTM/EnvironmentPert.py::ArpodCrtbp` | `(16,)` | `(3,)` | same |
| MLP constant-angle | `MLPconstAng/Environment.py::ArpodCrtbp` | `(18,)` | `(3,)` | same |
| LSTM constant-angle | `LSTMconstAng/Environment.py::ArpodCrtbp` | `(18,)` | `(3,)` | same |

### Are MLP and LSTM environments identical?

No. Byte-level and diff inspection show they are **not** identical copies.

Observed differences (non-exhaustive):

- Comment / formatting differences around `dyn_uncertainty` and corridor reward.
- LSTM nominal removes the unused `attitude_const` helper that remains (commented call) in MLP nominal.
- Reward-line spacing / minor literal formatting (`-30` vs `- 30`).

Policy choice (PPO MLP vs RecurrentPPO LSTM) is primarily selected in the **training scripts**, not by renaming the environment class.

### Nominal vs perturbed

`EnvironmentPert.py` is **not** a small noise-parameter toggle on the nominal CRTBP equations. Inspection shows a different relative dynamics formulation that includes additional disturbance terms consistent with a bicircular restricted four-body problem (BRFBP) contribution and a solar-radiation-pressure (SRP) acceleration model.

Unconfirmed without author input: whether every Monte Carlo “Pert” script intends this full disturbance model, or whether some filenames are historical.

### Constant-angle variants

Compared with nominal `(16,)` observation environments, constant-angle environments:

- Expand observation to `(18,)` by storing thrust components `T` instead of `||T||` only (plus reward history).
- Slice IVP state as `y0=x0[0:-5]` instead of `x0[0:-3]`.
- Add plume-impingement / Earth-in-FoV / high-velocity termination-style penalties in `get_reward`.
- Change some reward scales (for example docking bonus and log-distance weight).

`MLPconstAng/Environment.py` and `LSTMconstAng/Environment.py` differ by essentially one comment line.

## Experiment family table

| Variant | Policy / algorithm | Environment module | Training entrypoint(s) | Evaluation / Monte Carlo entrypoint(s) | Disturbance model | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| MLP nominal | `stable_baselines3.PPO` + `MlpPolicy` | `MLP/Environment.py` | `MLP/main.py`, `MLP/main2.py` | `MLP/MonteCarlo.py`, `MLP/MonteCarlo2.py` | Nominal CRTBP + thrust failure + small `dyn_uncertainty` noise | `main2` / `MonteCarlo2` are alternate saved-model / hyperparameter lineages |
| MLP transfer | PPO MLP | `MLP/Environment.py` | `MLP/mainTLextra.py` | Uses saved `ppo_mlpTLextra` (script-local) | Same as nominal env | Transfer-learning style load-then-finetune script |
| MLP perturbed eval | PPO MLP (loaded) | `MLP/EnvironmentPert.py` | (no dedicated train script found) | `MLP/MonteCarloPert.py`, `MLP/MonteCarloPert2.py` | BRFBP + SRP relative model in `EnvironmentPert` | Loads nominal-trained zip names such as `ppo_mlp` / `ppo_mlp2` |
| LSTM nominal | `sb3_contrib.RecurrentPPO` | `LSTM/Environment.py` | `LSTM/main.py`, `LSTM/main2.py`, `LSTM/main3.py` | `LSTM/MonteCarlo.py`, `MonteCarlo2.py`, `MonteCarlo3.py` | Nominal CRTBP family | Multiple saved checkpoint names (`ppo_recurrent*`) |
| LSTM transfer | RecurrentPPO | `LSTM/Environment.py` | `LSTM/mainTLextra.py` | Script-local load of `ppo_recurrentTLextra` | Same as LSTM nominal env | |
| LSTM perturbed eval | RecurrentPPO (loaded) | `LSTM/EnvironmentPert.py` | (no dedicated train script found) | `LSTM/MonteCarloPert.py`, `MonteCarloPert2.py` | BRFBP + SRP relative model | |
| MLP constant-angle | PPO MLP | `MLPconstAng/Environment.py` | `MLPconstAng/mainConst.py`, `mainTLconst.py` | `MLPconstAng/MonteCarlo.py` | Nominal CRTBP + expanded constraints | Obs dim 18 |
| LSTM constant-angle | RecurrentPPO | `LSTMconstAng/Environment.py` | `LSTMconstAng/mainConst.py`, `mainTLconst.py` | `LSTMconstAng/MonteCarlo.py` | Same as MLP const-angle env (near-duplicate file) | Obs dim 18 |
| Extra / classical | OCP (`mpopt`/`casadi`), LQR (`control`) | N/A / own dynamics | `ExtraCode/OCP*.py`, `ExtraCode/LQR.py` | `ExtraCode/OCPmcm*.py` | Problem-specific | Optional heavy deps; not required for RL smoke tests |
| Historical LSTM | RecurrentPPO | `ExtraCode/LSTM_old/Environment.py` | `ExtraCode/LSTM_old/main.py` | `ExtraCode/LSTM_old/MonteCarlo.py` | Older env (obs dim 14) | Preserved for history; not part of current smoke matrix |

## Callbacks and plotting

- `*/CallBack.py`: identical across MLP/LSTM/constAng families; SB3 `BaseCallback` that resets the env on rollout end. Importing it does not create a model.
- `*/PlotTensorboard.py` (+ `PlotTensorboard2.py` where present): read TensorBoard CSV exports and save PDF figures. Many scripts still hard-code the author’s local absolute path under `C:\Users\giova\PycharmProjects\...`. Relative CSV files also exist under each family `tensorboard/` folder.
- Training scripts write TensorBoard logs under `./tensorboard/` when run from their directory.

## Saved models and long runs

Training and Monte Carlo scripts reference local `.zip` checkpoints (for example `ppo_mlp01B`, `ppo_recurrentBest`). These artifacts are **not** guaranteed to be present in a fresh clone. Scripts that call `PPO.load(...)` / `RecurrentPPO.load(...)` will fail without the matching file.

Full training uses very large `total_timesteps` (on the order of `1e7` in several mains) and is intentionally **not** exercised by the reproducibility smoke tests.

## Import / working-directory notes

- Historical training, Monte Carlo, and plotting scripts still use bare imports such as `from Environment import ArpodCrtbp`. Running them typically requires the process working directory (or `sys.path`) to include that experiment directory.
- Smoke tests and `scripts/smoke_test_envs.py` load the six formal `Environment*.py` modules via `Path(__file__).resolve()` and `importlib`. They do not modify research entrypoints and do not call `os.chdir()`.
- Making historical entrypoints import-safe (`if __name__ == "__main__":` guards and `__file__`-based path bootstraps) is deferred to a follow-up maintenance PR.

## Currently unverified

- Which single script/checkpoint pair reproduces each quantitative claim in the associated thesis/paper.
- Whether `main` vs `main2` vs `main3` are successive improvements or parallel ablations.
- Authoritative naming of “perturbed” versus the BRFBP+SRP implementation details.
- Completeness of tracked pretrained model binaries in Git LFS / releases (none required for smoke tests).
