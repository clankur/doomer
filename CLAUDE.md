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
uv run python -m train --config-name=local_test +paths.model_name=smoke_000
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

## Architecture

- `train.py` — Network architecture, algorithm (REINFORCE/PPO/GRPO), training loop, Hydra entrypoint. The whole training step should be readable linearly in this file.
- `doom_env.py` — ViZDoom Gymnasium wrapper with preprocessing (grayscale, resize, normalize, frame stack). Provides `make_env` and `make_vec_env` factories.
- `configs/` — Hydra YAML configs. `base.yaml` is the schema; other configs inherit from it.

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
