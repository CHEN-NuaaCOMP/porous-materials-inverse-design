"""Export one canonical candidate to the legacy reconstruction CSV interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.io_utils import export_reconstruction_csv, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = read_jsonl(args.input)
    selected = next((r for r in records if r["candidate_id"] == args.candidate_id), None)
    if selected is None:
        raise KeyError(f"candidate not found: {args.candidate_id}")
    export_reconstruction_csv(selected, args.output)
    meta = Path(args.output).with_suffix(".metadata.json")
    meta.write_text(json.dumps({
        "candidate_id": selected["candidate_id"], "source": selected.get("source"),
        "units": selected.get("units"), "schema_version": selected.get("schema_version"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output} and {meta}")


if __name__ == "__main__":
    main()

