from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import tempfile
import time

_DEFAULT_EXP_ROOT = "../PancakeBot_var_exp"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name-prefix", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--summary-json", type=str, required=True)
    parser.add_argument("--cycle-audit-csv", type=str, default="var/runtime/dry_cycle_audit.csv")
    parser.add_argument("--output-dir", type=str, default=_DEFAULT_EXP_ROOT)
    parser.add_argument("--recent-cycles", type=int, default=12)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--duration-seconds", type=int, default=43_200)
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
    raise RuntimeError(f"profile_set_penalty_shadow_monitor_subprocess_failed:\n{tail_text[-4000:]}")


def _load_cycle_signature(path: Path) -> tuple[int, int | None]:
    if not path.exists():
        return (0, None)
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return (0, None)
    latest_epoch_raw = str(rows[-1].get("current_epoch", "")).strip()
    latest_epoch = int(latest_epoch_raw) if latest_epoch_raw else None
    return (int(len(rows)), latest_epoch)


def _validate_refresh_argv(*, python_exe: str, args: argparse.Namespace, output_dir: Path) -> list[str]:
    argv = [
        str(python_exe),
        "-m",
        "inspection.run_profile_set_penalty_shadow_validate_refresh",
        "--name-prefix",
        str(args.name_prefix),
        "--output-dir",
        str(output_dir),
        "--recent-cycles",
        str(int(args.recent_cycles)),
    ]
    if args.refresh_args:
        argv.extend(list(args.refresh_args))
    return argv


def main() -> None:
    args = _build_parser().parse_args()
    cwd = Path.cwd().resolve()
    python_exe = cwd / ".venv" / "Scripts" / "python.exe"
    if not python_exe.exists():
        raise FileNotFoundError(f"profile_set_penalty_shadow_monitor_python_missing: {python_exe}")
    output_dir = Path(str(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = Path(str(args.output_jsonl)).resolve()
    summary_json = Path(str(args.summary_json)).resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    cycle_audit_csv = Path(str(args.cycle_audit_csv)).resolve()
    validate_summary_json = output_dir / f"{args.name_prefix}_profile_set_penalty_shadow_validate_refresh_summary.json"

    start_ts = int(time.time())
    deadline_ts = int(start_ts) + int(args.duration_seconds)
    last_signature: tuple[int, int | None] | None = None

    while int(time.time()) < int(deadline_ts):
        signature = _load_cycle_signature(cycle_audit_csv)
        if signature != last_signature:
            _run_subprocess(
                argv=_validate_refresh_argv(
                    python_exe=str(python_exe),
                    args=args,
                    output_dir=output_dir,
                ),
                cwd=cwd,
            )
            validation = json.loads(validate_summary_json.read_text(encoding="utf-8"))
            record = {
                "monitor_ts": int(time.time()),
                "cycle_count": int(signature[0]),
                "latest_epoch": signature[1],
                **validation,
            }
            with output_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True))
                handle.write("\n")
            summary_json.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
            last_signature = signature
        time.sleep(max(1, int(args.poll_seconds)))


if __name__ == "__main__":
    main()
