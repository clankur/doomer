# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

doomer is a PyTorch RL training codebase using ViZDoom, for personal research. Algorithms: REINFORCE → PPO → GRPO (progressive implementation). The defining design value is **explicitness**: all visible in source rather than hidden behind abstractions. Preserve this when editing — prefer inline, readable code over new abstractions.

## Code Standards

### Verify reward improves

After any edit to `train.py`, run the local test config and confirm running reward increases over the ~50 episodes before declaring done.

### Comments explain _why_, not _what_

Leave a comment only when the next reader would otherwise be confused or repeat a known mistake.

## Common commands

### Install

Requires Python >= 3.11. CPU dev:

```
uv sync --extra cpu
```

For GPU use `uv sync --extra gpu`.

### Local CPU smoke test

```
RUNQ_EXPERIMENT_ID=local uv run python -m train --config-name=local_test ++paths.model_name=smoke_000
```

`RUNQ_EXPERIMENT_ID` tells `execute_remotely()` it's already on a worker, so training runs locally instead of submitting to the queue.

### Run on remote GPU (fractal)

```
uv run python -m train --config-name=local_test ++paths.model_name=my_experiment
```

Without `RUNQ_EXPERIMENT_ID`, this auto-captures the git context (repo, branch, commit, uncommitted diff), submits to the runq queue, and exits. The worker on fractal clones the repo, checks out the commit, applies the diff, runs `uv sync --extra gpu`, and trains.

Requires `RUNQ_SERVER` in your shell (`~/.zshrc`):

```
export RUNQ_SERVER="http://192.168.4.85:8080"
```

### Monitor experiments

- **Dashboard**: http://192.168.4.85:8080
- **WandB**: https://wandb.ai/clankur-personal/doomer
- **CLI**: `runq list`, `runq logs -f <id>`, `runq cancel <id>`

### Hydra overrides

Override any config value with `++key=value`:

```
uv run python -m train --config-name=local_test ++paths.model_name=ppo_lr3e4 ++training.learning_rate=3e-4
```

Each file in `configs/` lists its own intended launch command at the top — copy from there for real runs. `paths.model_name` controls the checkpoint subdirectory under `paths.root_working_dir` (default `/tmp`); change it per run.

### Lint and format

```
uvx ruff check
uvx ruff format
```

Ruff config lives in `pyproject.toml` (line-length 120, py310, rules `E4,E7,E9,F,I`).

### Tests

There is no test suite. The local test config above is the smoke test.

### Experiment reports (Quarto)

#### Style

When writing reports use the matx style which is: concise, impersonal, declarative, no qualifiers, math where needed, tables for data, zero speculation that isn't backed by a measurement. Every section does exactly one thing. For example you can review this blog post to review the content: 
[Future leakage in block-quantized attention](https://matx.com/research/leaky_quantization)


#### Usage

Reports live in `docs/` as `.qmd` files. Each report is an executable document mixing markdown, LaTeX equations, and Python analysis code (pandas, plotly) that renders to an interactive HTML page.

```
quarto preview docs/   # live-reload dev server
quarto render docs/    # build static site to docs/_site/
```

To add a new report: create `docs/<slug>.qmd`, then add a navbar entry in `docs/_quarto.yml` and a row in `docs/index.qmd`.

Publish the site to GitHub Pages with:

```
quarto publish gh-pages
```

This renders locally, pushes to the `gh-pages` branch, and opens the deployed site.

## Architecture

- `train.py` — Network architecture, algorithm (REINFORCE/PPO/GRPO), training loop, wandb logging, runq remote execution, Hydra entrypoint. The whole training step should be readable linearly in this file.
- `doom_env.py` — ViZDoom Gymnasium wrapper with preprocessing (grayscale, resize, normalize, frame stack). Provides `make_env` and `make_vec_env` factories.
- `configs/` — Hydra YAML configs. `base.yaml` is the schema; other configs inherit from it.
- `docs/` — Quarto experiment reports (`.qmd` files). `_quarto.yml` configures the site; rendered HTML goes to `docs/_site/` (gitignored).

## Coding patterns in `train.py`

### Minimal modules

Model, loss, optimizer, training loop, and Hydra entrypoint all live in `train.py`. Only natural utility boundaries (env wrapper, data loading) get their own files. Resist extracting helpers; the explicitness goal depends on a reader being able to see the whole step linearly.

### Config dataclasses

Use `@dataclass(frozen=True)` for all config/hparam records. Hydra loads YAML → `DictConfig`, then `build_config()` converts to typed dataclasses.

### Type annotations

All functions must have full type signatures — arguments and return types. Use dataclasses (e.g. `EpisodeResult`, `StepMetrics`) to bundle related return values instead of returning bare tuples. Annotate local variables where the type isn't obvious from the RHS.

### Einops for shape transforms

Use `einops.rearrange` instead of `.reshape`, `.unsqueeze`, `.squeeze`, `.permute`, `.view`. The pattern string documents what dimensions mean:

```python
# Good — dimensions are named and the transform is self-documenting
rearrange(obs, "frames h w -> 1 frames h w")
rearrange(x, "batch channels h w -> batch (channels h w)")
einsum(q, k, "batch heads seq_q dim, batch heads seq_k dim -> batch heads seq_q seq_k")

# Bad — opaque, reader must infer what dim 0 is
obs.unsqueeze(0)
x.reshape(x.size(0), -1)
torch.einsum("bhqd,bhkd->bhqk", q, k)
```

Prefer `einops.einsum` over `torch.einsum` — named dimensions read as documentation. Annotate intermediate shapes with inline comments where the tensor flows through multiple operations (e.g. after conv layers, after projections).

## Remote infrastructure

- **runq** — Self-hosted experiment queue at https://github.com/clankur/runq. Server runs on fractal (192.168.4.85:8080) as a systemd user service.
- **fractal** — GPU box (RTX 4090, Ubuntu 22.04, driver 570, CUDA 12.8). SSH alias `fractal`, user `clankur`.
- **wandb** — Metrics logged to `clankur-personal/doomer` project. Auth via `~/.netrc` on both laptop and fractal.
