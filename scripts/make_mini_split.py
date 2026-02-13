#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path


def sample_jsonl(src: Path, dst: Path, n: int, seed: int) -> int:
    with src.open("r", encoding="utf-8") as f:
        rows = [line for line in f if line.strip()]
    rng = random.Random(seed)
    n = min(n, len(rows))
    pick = rng.sample(rows, n) if n < len(rows) else rows
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        f.writelines(pick)
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Create mini train/val/test subsets from full jsonl splits.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_n", type=int, default=2000)
    parser.add_argument("--val_n", type=int, default=400)
    parser.add_argument("--test_n", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_n = sample_jsonl(in_dir / "train.jsonl", out_dir / "train.jsonl", args.train_n, args.seed)
    val_n = sample_jsonl(in_dir / "val.jsonl", out_dir / "val.jsonl", args.val_n, args.seed + 1)
    test_n = sample_jsonl(in_dir / "test.jsonl", out_dir / "test.jsonl", args.test_n, args.seed + 2)

    meta = {
        "source_dir": str(in_dir),
        "counts": {"train": train_n, "val": val_n, "test": test_n},
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"mini split written: {out_dir} | train={train_n} val={val_n} test={test_n}")


if __name__ == "__main__":
    main()
