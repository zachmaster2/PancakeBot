"""Restore kline data files after sync incorrectly trimmed them.

Merges:
- .bak files (original data, includes old invalid epochs)
- Current files (trimmed to 30k, but has 92 new klines)

Filters to only valid epochs (>= 437560, matching closed_rounds.jsonl).
"""
import json
from pathlib import Path


def load_klines(path: str) -> dict[int, dict]:
    """Load kline JSONL into {epoch: record}."""
    records = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        records[int(rec["epoch"])] = rec
    return records


def main():
    # Load closed rounds to get valid epoch set
    rounds_path = "var/closed_rounds.jsonl"
    round_epochs = set()
    for line in Path(rounds_path).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        round_epochs.add(int(rec["epoch"]))

    print(f"Closed rounds: {len(round_epochs)} epochs, range [{min(round_epochs)}..{max(round_epochs)}]")

    for label, current_path, bak_path in [
        ("BNB spot", "var/bnb_spot_prices.jsonl", "var/bnb_spot_prices.jsonl.bak"),
        ("BTC spot", "var/btc_spot_prices.jsonl", "var/btc_spot_prices.jsonl.bak"),
    ]:
        print(f"\n=== {label} ===")

        # Load both sources
        current = load_klines(current_path)
        bak = load_klines(bak_path)

        print(f"  Current: {len(current)} records, range [{min(current)}..{max(current)}]")
        print(f"  Backup:  {len(bak)} records, range [{min(bak)}..{max(bak)}]")

        # Merge: current takes precedence (has newest data)
        merged = {}
        for epoch, rec in bak.items():
            if epoch in round_epochs:
                merged[epoch] = rec
        for epoch, rec in current.items():
            if epoch in round_epochs:
                merged[epoch] = rec  # overwrite bak with current

        print(f"  Merged:  {len(merged)} records, range [{min(merged)}..{max(merged)}]")

        # Check alignment with rounds
        missing = round_epochs - set(merged.keys())
        extra = set(merged.keys()) - round_epochs
        print(f"  Missing (rounds without klines): {len(missing)}")
        print(f"  Extra (klines without rounds):   {len(extra)}")
        if missing:
            print(f"    Missing epochs: {sorted(missing)[:10]}...")

        # Write merged file
        sorted_records = sorted(merged.values(), key=lambda r: int(r["epoch"]))
        with open(current_path, "w", encoding="utf-8") as f:
            for rec in sorted_records:
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")

        print(f"  Written: {len(sorted_records)} records to {current_path}")


if __name__ == "__main__":
    main()
