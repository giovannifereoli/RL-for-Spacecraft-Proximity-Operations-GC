<p align="center">
  <img align="center" src="https://getvectorlogo.com/wp-content/uploads/2019/10/politecnico-di-milano-vector-logo.png" width="250" />
  <img align="center" src="https://www.colorado.edu/brand/sites/default/files/styles/large_image_style/public/page/boulder-one-line-reverse.png?itok=edXL_T9O" width="400" />
</p>

<div align="center">

![GitHub Repo stars](https://img.shields.io/github/stars/giovannifereoli/RL-for-Spacecraft-Proximity-Operations-GC?style=social)
![GitHub last commit](https://img.shields.io/github/last-commit/giovannifereoli/RL-for-Spacecraft-Proximity-Operations-GC)
[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.0.1-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![CI](https://github.com/giovannifereoli/RL-for-Spacecraft-Proximity-Operations-GC/actions/workflows/ci.yml/badge.svg)](https://github.com/giovannifereoli/RL-for-Spacecraft-Proximity-Operations-GC/actions/workflows/ci.yml)

</div>

## Meta-Reinforcement Learning for Spacecraft Proximity Operations Guidance and Control in Cislunar Space

In order to tackle the challenges of the future space exploration, new lightweight and model-free
guidance algorithms are needed to make spacecrafts autonomous. Indeed, in the last few decades
autonomous spacecraft guidance has become an active research topic and certainly in the next years
this technology will be needed to ensure proximity operation capabilities in the cislunar space. For
instance, NASA’s Artemis program plans to establish a lunar Gateway and this type of autonomous
manoeuvres, besides nominal rendezvous and docking (RV&D) ones, will be needed also for assembly
and maintenance procedures.

In this context a Meta-Reinforcement Learning (Meta-RL) algorithm will be applied to address the
real-time relative optimal guidance problem of a spacecraft in cislunar environment. Non-Keplerian
orbits have a more complex dynamics and classic control theory is less flexible and more
computationally expensive with respect to Machine Learning (ML) methods. Moreover, Meta-RL is
chosen for its elegant and promising ability of ‘‘learning how to learn’’ through experience.

A stochastic optimal control problem will be modelled in the Circular Restricted Three-Body Problem
(CRTBP) framework as a time-discrete Markov Decision Process (MDP). Then a Deep-RL agent,
composed by Long Short-Term Memory (LSTM) as Recurrent Neural Network (RNN), will be trained with
a state-of-the-art actor-critic algorithm known as Proximal Policy Optimization (PPO). In addition,
operational constraints and stochastic effects will be considered to assess solution safety and
robustness.

## Repository status

This repository now includes a **reproducibility baseline** that verifies:

- dependency installation on Python 3.9
- syntax checks for formal Python sources
- import + `reset`/`step` smoke tests for the six formal environment variants (MLP/LSTM nominal, perturbed, and constant-angle)

Maintenance scope intentionally does **not** re-run full thesis/paper PPO / RecurrentPPO training, does **not** regenerate Monte Carlo results, does **not** reconfirm every quantitative paper result, and does **not** migrate the stack to Gymnasium. Modular cleanup of historical training / Monte Carlo / plotting entrypoints (import side effects / main guards) is deferred to a follow-up PR.

## Installation

Python **3.9** is the verified target. Create and activate a virtual environment, then install bootstrap tooling required by `gym==0.21.0` (modern pip/packaging rejects that release’s metadata):

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux / macOS
# source .venv/bin/activate

python -m pip install "pip==23.1.2" "setuptools==65.5.0" "wheel==0.38.4" "packaging==21.3"
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

Optional CPU-only PyTorch install (recommended for CI / laptops without CUDA):

```bash
python -m pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

### Environment setup notes

- Formal RL code still uses the **original Gym API** (`gym==0.21.0`), not Gymnasium.
- Core verified stack: `numpy<2`, `scipy`, `matplotlib`, `pandas`, `torch==2.0.1`, `stable-baselines3==1.8.0`, `sb3-contrib==1.8.0`.
- `ExtraCode/` classical baselines may need extra packages (`casadi`, `mpopt`, `control`, `seaborn`) that are **not** part of the core requirements.

## Quick smoke test

From the repository root:

```bash
python scripts/smoke_test_envs.py
```

This instantiates the formal environment classes (MLP/LSTM nominal, perturbed, and constant-angle), runs `reset`, and takes a few `step` calls. It does not train policies or open GUI windows.

## Test suite

```bash
python -m compileall MLP LSTM MLPconstAng LSTMconstAng ExtraCode scripts tests
pytest -q
```

Tests are intentionally fast and do **not** run formal PPO / RecurrentPPO training, Monte Carlo batches, Weights & Biases, GPU jobs, or video generation.

## Experiment map

See [`docs/experiment-map.md`](docs/experiment-map.md) for the experiment lineage: MLP vs LSTM, nominal vs perturbed, constant-angle, transfer-learning scripts, Monte Carlo entrypoints, and known uncertainties.

## Main experiment families

| Family | Algorithm | Typical training entry | Typical Monte Carlo entry |
| --- | --- | --- | --- |
| MLP nominal | PPO (`MlpPolicy`) | `MLP/main.py`, `MLP/main2.py` | `MLP/MonteCarlo.py`, `MLP/MonteCarlo2.py` |
| MLP perturbed eval | PPO load + `EnvironmentPert` | (no dedicated train script found) | `MLP/MonteCarloPert.py`, `MLP/MonteCarloPert2.py` |
| LSTM nominal | RecurrentPPO | `LSTM/main.py`, `LSTM/main2.py`, `LSTM/main3.py` | `LSTM/MonteCarlo.py`, `MonteCarlo2.py`, `MonteCarlo3.py` |
| LSTM perturbed eval | RecurrentPPO load + `EnvironmentPert` | (no dedicated train script found) | `LSTM/MonteCarloPert.py`, `MonteCarloPert2.py` |
| MLP constant-angle | PPO | `MLPconstAng/mainConst.py` | `MLPconstAng/MonteCarlo.py` |
| LSTM constant-angle | RecurrentPPO | `LSTMconstAng/mainConst.py` | `LSTMconstAng/MonteCarlo.py` |

Historical training / Monte Carlo scripts use directory-local imports (for example `from Environment import ArpodCrtbp`). Run them from the corresponding family directory, for example:

```bash
cd MLP
python main.py
```

Saved `.zip` model files referenced by Monte Carlo scripts may be absent in a fresh clone. Making those entrypoints import-safe from the repository root is deferred to a follow-up maintenance PR.

## Notes on long-running training

- Several training entrypoints use `total_timesteps` on the order of `1e7`.
- Expect multi-hour (or longer) runs, TensorBoard logs under `./tensorboard/`, and large checkpoint writes.
- Plotting scripts under `PlotTensorboard*.py` may still contain author-local absolute CSV paths; relative CSVs also exist under each family’s `tensorboard/` directory.

## Current compatibility scope

- Verified: install + six-environment `reset`/`step` smoke tests on Python 3.9 with the pinned Gym/SB3 stack above; lightweight CI runs `compileall`, `pytest`, and `scripts/smoke_test_envs.py`.
- Not verified in this maintenance pass: full paper training reproduction, Monte Carlo regeneration, and Gymnasium migration.
- The formal code still uses the original Gym API; Gymnasium migration is deferred so that research definitions stay untouched here.
- Historical training / Monte Carlo / plotting entrypoints are mapped in `docs/experiment-map.md` but are not claimed to be import-safe yet.

## Credits

This project has been created by [Giovanni Fereoli](https://github.com/giovannifereoli) in 2023.
For any problem, clarification or suggestion, you can contact the author at [giovanni.fereoli@mail.polimi.it](mailto:giovanni.fereoli@mail.polimi.it).

## License

The package is under the [MIT](https://choosealicense.com/licenses/mit/) license.
