"""
build_leaderboard.py  [REF]

Parses every submission.json under records/track_10min_16mb/ and
writes leaderboard.csv next to this file.

Run from repo root:
    python analysis/atlas/build_leaderboard.py

Fields vary a lot across early vs late submissions (the format evolved).
This script normalizes to a minimal common set and leaves gaps as empty
strings rather than crashing. That's intentional - the data is messy,
normalizing it reveals where the record-keeping improved over time.
"""

import csv
import json
import pathlib

RECORDS_DIR = pathlib.Path("records/track_10min_16mb")
OUT_CSV = pathlib.Path("analysis/atlas/leaderboard.csv")

FIELDNAMES = [
    "folder",
    "date",
    "author",
    "name",
    "val_bpb",
    "val_bpb_std",
    "artifact_bytes",
    "seeds",
    "hardware",
    "quantization",
    "compression",
    "technique_summary",
]


def extract(folder: pathlib.Path) -> dict:
    json_path = folder / "submission.json"
    if not json_path.exists():
        return {}

    with open(json_path) as f:
        d = json.load(f)

    # artifact_bytes: prefer mean across seeds if available
    artifact_bytes = ""
    seed_results = d.get("seed_results", {})
    if seed_results:
        sizes = [v.get("artifact_bytes", 0) for v in seed_results.values() if v.get("artifact_bytes")]
        if sizes:
            artifact_bytes = int(sum(sizes) / len(sizes))
    if not artifact_bytes:
        artifact_bytes = d.get("bytes_total", d.get("artifact_bytes_mean", ""))

    # seeds: normalize to count or list
    seeds = d.get("seeds", "")
    if isinstance(seeds, list):
        seeds = ";".join(str(s) for s in seeds)

    # technique_summary: use explicit field, or join techniques list, or blurb truncated
    tech = (
        d.get("technique_summary")
        or (", ".join(d["techniques"]) if "techniques" in d else "")
        or d.get("blurb", "")[:120]
    )

    return {
        "folder": folder.name,
        "date": d.get("date", "")[:10],  # strip time if present
        "author": d.get("author", ""),
        "name": d.get("name", ""),
        "val_bpb": d.get("val_bpb", ""),
        "val_bpb_std": d.get("val_bpb_std", ""),
        "artifact_bytes": artifact_bytes,
        "seeds": seeds,
        "hardware": d.get("hardware", ""),
        "quantization": d.get("quantization", ""),
        "compression": d.get("compression", ""),
        "technique_summary": tech,
    }


def main():
    rows = []
    for folder in sorted(RECORDS_DIR.iterdir()):
        if not folder.is_dir():
            continue
        row = extract(folder)
        if row and row.get("val_bpb"):
            rows.append(row)

    # sort by val_bpb ascending (lower is better)
    rows.sort(key=lambda r: float(r["val_bpb"]))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {OUT_CSV}")
    for r in rows[:5]:
        print(f"  {r['val_bpb']:.5f}  {r['name'][:60]}")


if __name__ == "__main__":
    main()
