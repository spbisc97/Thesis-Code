# Repository Guidelines

## Project Structure & Module Organization
- `Python/SafeGym/`: your Gymnasium-compatible package `safegym` (environments in `safegym/envs/`, assets in `envs/assets/`, tests in `Python/SafeGym/tests/`, packaging via `setup.py`).
- `Python/Test_env/`: SE2 docking experiments and notebooks (e.g., `satellite_se2_ppo.py`). Saved runs live in the `Python/Test_env/savings` submodule.
- `Matlab/Sat_SE2`, `Matlab/Sat_SE3`: MATLAB dynamics/Simulink models for RPO; entry points like `Tester.m`.
- Root: `environment.yml` (conda), `.gitmodules` (submodules), `README.md`.

## Build, Test, and Development Commands
- Environment: `conda env create -f environment.yml --prefix ./venv && conda activate ./venv`.
- Submodules: `git submodule update --init --recursive` (required for `SafeGym` and savings).
- Install SafeGym (editable): `pip install -e Python/SafeGym`.
- Run tests: `cd Python/SafeGym && pytest -q` (focus: `Satellite_SE2`).
- Quick run: `python Python/Test_env/satellite_se2_ppo.py` or `gym.make("Satellite-SE2-v0")`.

## Coding Style & Naming Conventions
- Python: PEP 8, 4-space indentation; modules/functions `snake_case`, classes `CapWords`. Keep existing public API (e.g., `Satellite_SE2`).
- Env registration: add new IDs in `safegym/envs/__init__.py` (pattern: `Name-v0`).
- Notebooks: keep minimal; move reusable code to modules in `safegym/` or `Test_env/`.

## Testing Guidelines
- Framework: `pytest`; tests in `Python/SafeGym/tests/` named `test_*.py`.
- Cover: `reset/step` contract, termination/truncation, determinism under fixed seed, `render_mode` behavior.
- Examples: `pytest -k Satellite_SE2 -q` to run SE2-only tests.

## Experiment Reproducibility (Paper)
- SE2 Docking: use `Satellite-SE2-v0` for Safe RL RPO experiments.
- Seeds & configs: record `seed`, `env kwargs`, and algorithm hyperparams; persist results under `Python/Test_env/savings/<algo>_<seed>_<date>/`.
- Artifacts: save learning curves, evaluation metrics, and key frames to `safegym/docs/` or the run folder; include a short `README.md` per run.

## Commit & Pull Request Guidelines
- Commits: imperative and specific (e.g., `env(se2): fix termination check`, `exp: add PPO config`), not “sync”.
- PRs: state goal, changes, test plan (commands), and attach plots/screenshots for docking runs; link issues.
- Gate: create/activate env, `pip install -e Python/SafeGym`, run `pytest -q` before submitting.

## Security & Configuration Tips
- Submodules are SSH; switch to HTTPS in `.gitmodules` if collaborators lack keys.
- Avoid committing large binaries; prefer links or generated assets under `docs/`.
