#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

import torch


ACTIONS = [
    "opening a door",
    "sitting on a chair",
    "drinking water",
    "reading a book",
    "writing on paper",
    "washing dishes",
    "cutting vegetables",
    "typing on laptop",
    "talking on phone",
    "running outside",
    "throwing a ball",
    "lifting a box",
]

STAGES = [
    "starts the action",
    "continues the motion",
    "interacts with object",
    "changes body pose",
    "completes the event",
    "reacts to outcome",
    "moves to next step",
    "finishes sequence",
]


def event_prompts(action: str, n_frames: int) -> list[str]:
    prompts = []
    for t in range(n_frames):
        stage = STAGES[t % len(STAGES)]
        prompts.append(f"The person is {action} and {stage}.")
    return prompts


def frame_prompts(action: str, n_frames: int) -> list[str]:
    return [f"A frame shows a person {action}." for _ in range(n_frames)]


def build_split(
    out_file: Path,
    n_samples: int,
    n_frames: int,
    teacher_bank: torch.Tensor,
    rng: random.Random,
) -> None:
    with out_file.open("w", encoding="utf-8") as f:
        for i in range(n_samples):
            action_id = rng.randrange(len(ACTIONS))
            action = ACTIONS[action_id]
            negatives = [x for idx, x in enumerate(ACTIONS) if idx != action_id]
            wrong = rng.sample(negatives, k=4)
            choices = [action] + wrong
            rng.shuffle(choices)
            answer_idx = choices.index(action)
            teacher_hard_idx = answer_idx if rng.random() > 0.1 else rng.randrange(len(choices))

            teacher_feat = teacher_bank[action_id] + torch.randn_like(teacher_bank[action_id]) * 0.03
            record = {
                "id": i,
                "dataset": "toy",
                "action_id": action_id,
                "question": "What is the main event in this video?",
                "choices": choices,
                "answer_idx": answer_idx,
                "teacher_hard_idx": teacher_hard_idx,
                "event_prompts": event_prompts(action, n_frames=n_frames),
                "frame_prompts": frame_prompts(action, n_frames=n_frames),
                "teacher_embedding": teacher_feat.tolist(),
                "frame_seed": rng.randrange(1, 1_000_000),
            }
            f.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create toy data for CaTeR minimal reproduction.")
    parser.add_argument("--out_dir", type=str, default="data/toy")
    parser.add_argument("--train_size", type=int, default=512)
    parser.add_argument("--val_size", type=int, default=128)
    parser.add_argument("--test_size", type=int, default=128)
    parser.add_argument("--num_frames", type=int, default=6)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    teacher_bank = torch.randn(len(ACTIONS), args.d_model) * 0.5

    build_split(out_dir / "train.jsonl", args.train_size, args.num_frames, teacher_bank, rng)
    build_split(out_dir / "val.jsonl", args.val_size, args.num_frames, teacher_bank, rng)
    build_split(out_dir / "test.jsonl", args.test_size, args.num_frames, teacher_bank, rng)
    print(f"Toy data written to: {out_dir}")


if __name__ == "__main__":
    main()
