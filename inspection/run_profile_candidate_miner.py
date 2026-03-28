from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import tempfile
import tomllib

from pancakebot.core.errors import InvariantError

_DEFAULT_EXP_ROOT = "../PancakeBot_var_exp"


@dataclass(frozen=True, slots=True)
class CandidateScore:
    profile_name: str
    active_candidate_name: str
    static_mean_per_500: float
    static_mean_bet_rate: float
    positive_window_count: int
    max_positive_streak: int
    positive_to_positive_rate: float
    distinct_positive_win_count: int
    skip_replacement_count: int
    marginal_oracle_gain_per_500: float
    marginal_skip_replacement_gain_per_500: float
    current_pool_skip_window_count: int


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.toml")
    parser.add_argument("--current-pool-compare-csv", type=str, required=True)
    parser.add_argument("--name-prefix", type=str, required=True)
    parser.add_argument("--window-size-rounds", type=int, default=216)
    parser.add_argument("--num-windows", type=int, default=20)
    parser.add_argument("--source-tail-rounds", type=int, default=30000)
    parser.add_argument("--initial-bankroll-bnb", type=float, default=50.0)
    parser.add_argument("--exclude-candidate-names", type=str, default="disloc_stageB_bullonly_recent8pct_v1,disloc_stageG2_bullonly_recent5pct_v1")
    parser.add_argument("--candidate-name", action="append", default=[])
    parser.add_argument("--output-dir", type=str, default=_DEFAULT_EXP_ROOT)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def _parse_name_list(raw: str) -> list[str]:
    text = str(raw).strip()
    if text == "":
        return []
    return [str(token).strip() for token in text.split(",") if str(token).strip() != ""]


def _load_dislocation_candidate_names(config_path: Path) -> list[str]:
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)
    candidates = (
        data.get("strategy", {})
        .get("dislocation", {})
        .get("candidates", {})
    )
    if not isinstance(candidates, list) or not candidates:
        raise InvariantError("profile_candidate_miner_dislocation_candidates_missing")
    out: list[str] = []
    for section in candidates:
        if not isinstance(section, dict):
            continue
        name = str(section.get("name", "")).strip()
        if name != "":
            out.append(str(name))
    if not out:
        raise InvariantError("profile_candidate_miner_dislocation_candidate_names_empty")
    return out


def _alias_from_candidate_name(active_candidate_name: str) -> str:
    text = str(active_candidate_name).strip()
    if text.startswith("disloc_"):
        return str(text[len("disloc_") :])
    return str(text)


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
    raise RuntimeError(f"profile_candidate_miner_subprocess_failed:\n{tail_text[-4000:]}")


def _load_compare_rows(compare_csv: Path) -> tuple[list[str], list[dict[str, float]], list[int]]:
    if not compare_csv.exists():
        raise FileNotFoundError(f"profile_candidate_miner_compare_missing: {compare_csv}")
    rows: list[dict[str, float]] = []
    offsets: list[int] = []
    profile_names: list[str] = []
    with compare_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise InvariantError("profile_candidate_miner_compare_header_missing")
        fieldnames = [str(x) for x in reader.fieldnames]
        profile_names = [
            str(name[: -len("_per_500")])
            for name in fieldnames
            if str(name).endswith("_per_500") and str(name) != "tail_offset_rounds"
        ]
        for raw in reader:
            offsets.append(int(raw["tail_offset_rounds"]))
            row: dict[str, float] = {}
            for name in profile_names:
                row[f"{name}_per_500"] = float(raw[f"{name}_per_500"])
                row[f"{name}_bet_rate"] = float(raw[f"{name}_bet_rate"])
            rows.append(row)
    return profile_names, rows, offsets


def _current_pool_best_with_skip(row: dict[str, float], profile_names: list[str]) -> float:
    return max([0.0] + [float(row[f"{name}_per_500"]) for name in profile_names])


def _max_positive_streak(values: list[float]) -> int:
    best = 0
    cur = 0
    for value in values:
        if float(value) > 0.0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _positive_to_positive_rate(values: list[float]) -> float:
    starts = 0
    cont = 0
    for idx in range(len(values) - 1):
        if float(values[idx]) > 0.0:
            starts += 1
            if float(values[idx + 1]) > 0.0:
                cont += 1
    if int(starts) <= 0:
        return 0.0
    return float(cont) / float(starts)


def _score_candidate(
    *,
    profile_name: str,
    active_candidate_name: str,
    candidate_rows: list[dict[str, float]],
    current_rows: list[dict[str, float]],
    current_profile_names: list[str],
) -> CandidateScore:
    per_500_values = [float(row[f"{profile_name}_per_500"]) for row in candidate_rows]
    bet_rates = [float(row[f"{profile_name}_bet_rate"]) for row in candidate_rows]
    marginal_gains: list[float] = []
    skip_replacement_gains: list[float] = []
    distinct_positive_win_count = 0
    skip_replacement_count = 0
    current_skip_window_count = 0
    for current_row, candidate_row in zip(current_rows, candidate_rows, strict=True):
        current_best = _current_pool_best_with_skip(current_row, current_profile_names)
        candidate_value = float(candidate_row[f"{profile_name}_per_500"])
        candidate_best = max(float(current_best), max(0.0, float(candidate_value)))
        marginal_gains.append(float(candidate_best - current_best))
        if float(current_best) <= 0.0:
            current_skip_window_count += 1
            if float(candidate_value) > 0.0:
                skip_replacement_count += 1
                skip_replacement_gains.append(float(candidate_value))
        if float(candidate_value) > 0.0:
            current_best_noskip = max(float(current_row[f"{name}_per_500"]) for name in current_profile_names)
            if float(candidate_value) > float(current_best_noskip):
                distinct_positive_win_count += 1
    return CandidateScore(
        profile_name=str(profile_name),
        active_candidate_name=str(active_candidate_name),
        static_mean_per_500=float(sum(per_500_values) / len(per_500_values)),
        static_mean_bet_rate=float(sum(bet_rates) / len(bet_rates)),
        positive_window_count=int(sum(1 for value in per_500_values if float(value) > 0.0)),
        max_positive_streak=_max_positive_streak(per_500_values),
        positive_to_positive_rate=_positive_to_positive_rate(per_500_values),
        distinct_positive_win_count=int(distinct_positive_win_count),
        skip_replacement_count=int(skip_replacement_count),
        marginal_oracle_gain_per_500=float(sum(marginal_gains) / len(marginal_gains)),
        marginal_skip_replacement_gain_per_500=(
            float(sum(skip_replacement_gains) / len(current_rows)) if current_rows else 0.0
        ),
        current_pool_skip_window_count=int(current_skip_window_count),
    )


def main() -> None:
    args = _build_parser().parse_args()
    cwd = Path.cwd().resolve()
    python_exe = cwd / ".venv" / "Scripts" / "python.exe"
    if not python_exe.exists():
        raise FileNotFoundError(f"profile_candidate_miner_python_missing: {python_exe}")
    config_path = Path(str(args.config)).resolve()
    current_compare_csv = Path(str(args.current_pool_compare_csv)).resolve()
    output_dir = Path(str(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    current_profile_names, current_rows, current_offsets = _load_compare_rows(current_compare_csv)
    if "stageb" not in current_profile_names:
        raise InvariantError("profile_candidate_miner_current_pool_missing_stageb")

    candidate_names = [str(x) for x in list(args.candidate_name)]
    if not candidate_names:
        candidate_names = _load_dislocation_candidate_names(config_path)
    exclude_names = set(_parse_name_list(str(args.exclude_candidate_names)))
    candidate_names = [str(name) for name in candidate_names if str(name) not in exclude_names]
    if not candidate_names:
        raise InvariantError("profile_candidate_miner_candidate_names_empty_after_exclude")

    mine_prefix = f"{args.name_prefix}_mine"
    argv = [
        str(python_exe),
        "-m",
        "inspection.run_profile_set_window_selector",
        "--config",
        str(config_path),
        "--name-prefix",
        str(mine_prefix),
        "--window-size-rounds",
        str(int(args.window_size_rounds)),
        "--tail-offset-rounds",
        ",".join(str(int(x)) for x in current_offsets),
        "--source-tail-rounds",
        str(int(args.source_tail_rounds)),
        "--initial-bankroll-bnb",
        str(float(args.initial_bankroll_bnb)),
        "--selector-lookbacks",
        "1,2,3,4,5",
        "--selector-margins-per-500=-0.2,0.0,0.2,0.5",
        "--selector-skip-thresholds-per-500",
        "0.0,0.05,0.1",
        "--min-selected-bet-rate",
        "0.05",
    ]
    if bool(args.no_resume):
        argv.append("--no-resume")
    alias_to_candidate: dict[str, str] = {}
    for active_candidate_name in candidate_names:
        alias = _alias_from_candidate_name(str(active_candidate_name))
        alias_to_candidate[str(alias)] = str(active_candidate_name)
        argv.extend(
            [
                "--dislocation-profile",
                f"name={alias},active_candidate_name={active_candidate_name}",
            ]
        )
    _run_subprocess(argv=argv, cwd=cwd)

    mined_compare_csv = output_dir / f"{mine_prefix}_profile_set_window_compare.csv"
    mined_profile_names, mined_rows, mined_offsets = _load_compare_rows(mined_compare_csv)
    if current_offsets != mined_offsets:
        raise InvariantError("profile_candidate_miner_offset_mismatch")

    scores: list[CandidateScore] = []
    for profile_name in mined_profile_names:
        if str(profile_name) == "stageb":
            continue
        scores.append(
            _score_candidate(
                profile_name=str(profile_name),
                active_candidate_name=str(alias_to_candidate[str(profile_name)]),
                candidate_rows=mined_rows,
                current_rows=current_rows,
                current_profile_names=current_profile_names,
            )
        )
    scores = sorted(
        scores,
        key=lambda row: (
            int(row.skip_replacement_count),
            float(row.marginal_oracle_gain_per_500),
            int(row.distinct_positive_win_count),
            int(row.max_positive_streak),
            float(row.positive_to_positive_rate),
            float(row.static_mean_per_500),
        ),
        reverse=True,
    )
    csv_path = output_dir / f"{args.name_prefix}_profile_candidate_miner.csv"
    json_path = output_dir / f"{args.name_prefix}_profile_candidate_miner.json"
    summary_path = output_dir / f"{args.name_prefix}_profile_candidate_miner_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(scores[0]).keys()))
        writer.writeheader()
        for row in scores:
            writer.writerow(asdict(row))
    json_path.write_text(
        json.dumps([asdict(row) for row in scores], indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    summary = {
        "name_prefix": str(args.name_prefix),
        "current_pool_compare_csv": str(current_compare_csv),
        "mined_compare_csv": str(mined_compare_csv),
        "candidate_count": int(len(scores)),
        "top_candidates": [asdict(row) for row in scores[: min(5, len(scores))]],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
