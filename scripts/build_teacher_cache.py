#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cater.teachers import build_teacher


def main() -> None:
    parser = argparse.ArgumentParser(description="Build real teacher cache for VideoQA samples.")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--teacher_name", type=str, required=True, choices=["videollava", "flamingo", "chatunivl"])
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_id", type=str, default="LanguageBind/Video-LLaVA-7B-hf")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--num_video_frames", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--load_in_4bit", action="store_true")
    args = parser.parse_args()

    teacher = build_teacher(
        teacher_name=args.teacher_name,
        d_model=args.d_model,
        seed=args.seed,
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        num_video_frames=args.num_video_frames,
        max_new_tokens=args.max_new_tokens,
        load_in_4bit=args.load_in_4bit,
    )

    in_path = Path(args.input_jsonl)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    with out_path.open("w", encoding="utf-8") as f:
        for row in tqdm(rows, desc=f"teacher={args.teacher_name}"):
            out = teacher.infer(row)
            row["teacher_name"] = args.teacher_name
            row["teacher_embedding"] = out.embedding.tolist()
            row["teacher_hard_idx"] = int(out.hard_idx)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
