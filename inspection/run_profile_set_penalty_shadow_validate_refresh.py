from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

_DEFAULT_EXP_ROOT = "../PancakeBot_var_exp"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name-prefix", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=_DEFAULT_EXP_ROOT)
    parser.add_argument("--recent-cycles", type=int, default=12)
    parser.add_argument("--refresh-args", nargs=argparse.REMAINDER)
    return parser


def _run_subprocess(*, argv: list[str], cwd: Path) -> None:
    with tempfile.NamedTemporaryFile(mode="w+b", suffix=".log", delete=False) as handle:
        log_path = Path(handle.name)
        with log_path.open("wb") as log_handle:
            completed = subprocess.run(
                argv,
                cwd=str(cwd),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
    if int(completed.returncode) == 0:
        try:
            log_path.unlink(missing_ok=True)
        except Exception:
            pass
        return
    try:
        tail_text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        tail_text = ""
    raise RuntimeError(f"profile_set_penalty_shadow_validate_refresh_subprocess_failed:\n{tail_text[-4000:]}")


def _refresh_argv(*, python_exe: str, args: argparse.Namespace, output_dir: Path) -> list[str]:
    argv = [
        str(python_exe),
        "-m",
        "inspection.run_profile_set_penalty_shadow_refresh",
        "--name-prefix",
        str(args.name_prefix),
        "--output-dir",
        str(output_dir),
    ]
    if args.refresh_args:
        argv.extend(list(args.refresh_args))
    return argv


def _validation_argv(*, python_exe: str, args: argparse.Namespace, output_dir: Path) -> list[str]:
    recommendation_json = output_dir / f"{args.name_prefix}_profile_set_penalty_shadow_recommendation.json"
    return [
        str(python_exe),
        "-m",
        "inspection.run_profile_set_shadow_validation",
        "--recommendation-json",
        str(recommendation_json),
        "--name-prefix",
        str(args.name_prefix),
        "--output-dir",
        str(output_dir),
        "--recent-cycles",
        str(int(args.recent_cycles)),
    ]


def main() -> None:
    args = _build_parser().parse_args()
    cwd = Path.cwd().resolve()
    python_exe = cwd / ".venv" / "Scripts" / "python.exe"
    if not python_exe.exists():
        raise FileNotFoundError(f"profile_set_penalty_shadow_validate_refresh_python_missing: {python_exe}")
    output_dir = Path(str(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    refresh_argv = _refresh_argv(
        python_exe=str(python_exe),
        args=args,
        output_dir=output_dir,
    )
    _run_subprocess(argv=refresh_argv, cwd=cwd)
    recommendation_json = output_dir / f"{args.name_prefix}_profile_set_penalty_shadow_recommendation.json"
    validation_argv = _validation_argv(
        python_exe=str(python_exe),
        args=args,
        output_dir=output_dir,
    )
    _run_subprocess(argv=validation_argv, cwd=cwd)
    validation_summary_path = output_dir / f"{args.name_prefix}_profile_set_shadow_validation_summary.json"
    validation_summary = json.loads(validation_summary_path.read_text(encoding="utf-8"))
    summary = {
        "name_prefix": str(args.name_prefix),
        "recommendation_json": str(recommendation_json),
        "validation_summary_json": str(validation_summary_path),
        "shadow_chosen_profile": str(validation_summary.get("shadow_chosen_profile", "")),
        "shadow_chosen_predicted_per_500": float(validation_summary.get("shadow_chosen_predicted_per_500", 0.0)),
        "coherence_status": str(validation_summary.get("coherence_status", "")),
        "coherence_reason": str(validation_summary.get("coherence_reason", "")),
    }
    summary_path = output_dir / f"{args.name_prefix}_profile_set_penalty_shadow_validate_refresh_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
