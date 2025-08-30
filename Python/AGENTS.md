# Repository Guidelines

## Project Structure & Module Organization
- `Python/SafeGym/`: Gymnasium package `safegym` (envs in `safegym/envs/`, assets in `envs/assets/`, tests in `Python/SafeGym/tests/`, packaging via `setup.py`).
- `Python/Test_env/`: SE2 docking experiments and notebooks (e.g., `satellite_se2_ppo.py`). Saved runs under `Python/Test_env/savings/` (git submodule).
- `Matlab/Sat_SE2`, `Matlab/Sat_SE3`: MATLAB dynamics/Simulink models (entry points like `Tester.m`).
- Root: `environment.yml`, `.gitmodules`, `README.md`.

## Build, Test, and Development Commands
- Create/activate env: `conda env create -f environment.yml --prefix ./venv && conda activate ./venv`.
- Init submodules: `git submodule update --init --recursive` (required for `SafeGym` and `savings`).
- Install SafeGym (editable): `pip install -e Python/SafeGym`.
- Run tests: `cd Python/SafeGym && pytest -q` (SE2 only: `pytest -k Satellite_SE2 -q`).
- Quick run: `python Python/Test_env/satellite_se2_ppo.py` or `gym.make("Satellite-SE2-v0")` in Python.

## Coding Style & Naming Conventions
- Python: PEP 8, 4-space indentation; modules/functions `snake_case`, classes `CapWords`.
- Keep public API stable (e.g., `Satellite_SE2`).
- Env registration in `safegym/envs/__init__.py` using `Name-v0` IDs.
- Keep notebooks minimal; move reusable code to `safegym/` or `Test_env/`.

## Testing Guidelines
- Framework: `pytest`; tests live in `Python/SafeGym/tests/` and are named `test_*.py`.
- Cover: `reset/step` contract, termination/truncation, determinism under fixed seeds, `render_mode` behavior.
- Example: `pytest -k Satellite_SE2 -q`.

## Experiment Reproducibility
- Use `Satellite-SE2-v0` for Safe RL RPO experiments.
- Record `seed`, env kwargs, and algorithm hyperparams; save to `Python/Test_env/savings/<algo>_<seed>_<date>/`.
- Store learning curves, metrics, and key frames; include a short run `README.md`.

## Commit & Pull Request Guidelines
- Commits: imperative and specific (e.g., `env(se2): fix termination check`, `exp: add PPO config`).
- PRs: state goal, summarize changes, include test plan (commands), link issues, and attach plots/screenshots for docking runs.
- Gate: create/activate env, `pip install -e Python/SafeGym`, run `pytest -q` before submitting.

## Security & Configuration Tips
- Submodules default to SSH; use HTTPS in `.gitmodules` if collaborators lack keys.
- Do not commit large binaries; prefer links or generated assets under `safegym/docs/` or run folders.
