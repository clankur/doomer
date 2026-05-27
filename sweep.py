"""Submit a hyperparameter sweep to runq.

Usage:
    uv run python sweep.py training.learning_rate=1e-4,3e-4,1e-3 training.gamma=0.95,0.99
    uv run python sweep.py --dry-run training.entropy_coeff=0.01,0.05,0.1
    uv run python sweep.py --config-name=base --max-episodes=500 training.learning_rate=1e-4,3e-4
"""

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

ABBREVS: dict[str, str] = {
    "learning_rate": "lr",
    "entropy_coeff": "ec",
    "value_loss_coeff": "vc",
    "clip_epsilon": "clip",
    "gae_lambda": "lam",
    "ppo_epochs": "epochs",
    "minibatch_size": "mb",
}


def abbreviate_key(dotted_key: str) -> str:
    last = dotted_key.rsplit(".", maxsplit=1)[-1]
    return ABBREVS.get(last, last)


def parse_sweep_spec(spec: str) -> tuple[str, list[str]]:
    key, _, vals = spec.partition("=")
    if not key or not vals:
        print(f"error: invalid sweep spec '{spec}', expected key=v1,v2,...", file=sys.stderr)
        sys.exit(1)
    return key, vals.split(",")


def build_model_name(combo: list[tuple[str, str]]) -> str:
    parts = [f"{abbreviate_key(key)}{val}" for key, val in combo]
    return "sweep_" + "_".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit hyperparameter sweep to runq")
    parser.add_argument("specs", nargs="+", metavar="key=v1,v2,...", help="Sweep specs")
    parser.add_argument("--config-name", default="local_test", help="Hydra config (default: local_test)")
    parser.add_argument("--max-episodes", type=int, default=None, help="Override training.max_episodes")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without submitting")
    args = parser.parse_args()

    sweep_params: list[tuple[str, list[str]]] = [parse_sweep_spec(s) for s in args.specs]
    keys = [key for key, _ in sweep_params]
    value_lists = [vals for _, vals in sweep_params]

    combos = list(itertools.product(*value_lists))
    print(f"{len(combos)} runs from {' x '.join(str(len(v)) for v in value_lists)} grid")

    project_dir = Path(__file__).parent
    submitted = 0

    for values in combos:
        combo = list(zip(keys, values))
        model_name = build_model_name(combo)

        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "train",
            f"--config-name={args.config_name}",
            f"++paths.model_name={model_name}",
        ]
        for key, val in combo:
            cmd.append(f"++{key}={val}")
        if args.max_episodes is not None:
            cmd.append(f"++training.max_episodes={args.max_episodes}")

        if args.dry_run:
            print(" ".join(cmd))
        else:
            result = subprocess.run(cmd, cwd=project_dir)
            if result.returncode == 0:
                submitted += 1
            else:
                print(f"warning: {model_name} exited with code {result.returncode}", file=sys.stderr)

    if not args.dry_run:
        print(f"\n{submitted}/{len(combos)} runs submitted to runq")


if __name__ == "__main__":
    main()
