#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cater import CaTeRMini, HashTokenizer, VideoQADataset, collate_batch


@torch.no_grad()
def evaluate(model: CaTeRMini, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0
    correct = 0
    for batch in loader:
        frames = batch["frames"].to(device)
        prompt_ids = batch["prompt_ids"].to(device)
        choice_ids = batch["choice_ids"].to(device)
        choice_mask = batch["choice_mask"].to(device)
        labels = batch["answer_idx"].to(device)

        logits, _ = model(frames=frames, prompt_ids=prompt_ids, choice_ids=choice_ids, choice_mask=choice_mask)
        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
    return correct / max(1, total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CaTeR reproduction model.")
    parser.add_argument("--data_dir", type=str, default="data/toy")
    parser.add_argument("--split_file", type=str, default="test.jsonl")
    parser.add_argument("--checkpoint", type=str, default="runs/cater_repro/best.pt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["args"]

    tokenizer = HashTokenizer()
    dataset = VideoQADataset(
        jsonl_path=Path(args.data_dir) / args.split_file,
        tokenizer=tokenizer,
        num_frames=cfg["num_frames"],
        image_size=cfg["image_size"],
        prompt_len=cfg["prompt_len"],
        choice_len=cfg["choice_len"],
        prompt_variant=cfg["prompt_variant"],
        teacher_dim=cfg["d_model"],
        seed=cfg.get("seed", 42) + 7,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CaTeRMini(
        d_model=cfg["d_model"],
        vocab_size=cfg["vocab_size"],
        image_size=cfg["image_size"],
        max_frames=cfg["num_frames"],
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)

    acc = evaluate(model, loader, device=device)
    print(f"split_file={args.split_file} top1_acc={acc:.4f}")


if __name__ == "__main__":
    main()
