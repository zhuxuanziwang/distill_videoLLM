#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cater import CLIPTokenizerWrapper, CaTeRMini, HashTokenizer, VideoQADataset, collate_batch


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _param_count(module: torch.nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def _safe_grad_norm(param: torch.nn.Parameter | None) -> float:
    if param is None or param.grad is None:
        return 0.0
    return float(param.grad.norm().item())


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
    parser = argparse.ArgumentParser(description="Train CaTeR reproduction with distillation/ablation support.")
    parser.add_argument("--data_dir", type=str, default="data/toy")
    parser.add_argument("--train_file", type=str, default="train.jsonl")
    parser.add_argument("--val_file", type=str, default="val.jsonl")
    parser.add_argument("--dataset_name", type=str, default="toy", choices=["toy", "tvqa", "activitynet_qa", "msrvtt_qa"])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_frames", type=int, default=6)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--prompt_len", type=int, default=32)
    parser.add_argument("--choice_len", type=int, default=64)
    parser.add_argument("--tokenizer", type=str, default="clip", choices=["clip", "hash"])
    parser.add_argument("--clip_model_id", type=str, default="openai/clip-vit-base-patch16")
    parser.add_argument("--prompt_variant", type=str, default="event", choices=["event", "frame", "random", "none"])
    parser.add_argument("--distill_mode", type=str, default="hybrid", choices=["none", "hard", "soft", "hybrid"])
    parser.add_argument("--lambda_soft", type=float, default=0.4)
    parser.add_argument("--lambda_hard", type=float, default=0.5)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="runs/cater_repro")
    parser.add_argument("--debug_model_prints", action="store_true")
    parser.add_argument("--debug_forward_print_steps", type=int, default=1)
    parser.add_argument("--debug_grad_print_steps", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.tokenizer == "clip":
        tokenizer = CLIPTokenizerWrapper(model_id=args.clip_model_id)
    else:
        tokenizer = HashTokenizer()
    train_set = VideoQADataset(
        jsonl_path=Path(args.data_dir) / args.train_file,
        tokenizer=tokenizer,
        num_frames=args.num_frames,
        image_size=args.image_size,
        prompt_len=args.prompt_len,
        choice_len=args.choice_len,
        prompt_variant=args.prompt_variant,
        teacher_dim=args.d_model,
        seed=args.seed,
    )
    val_set = VideoQADataset(
        jsonl_path=Path(args.data_dir) / args.val_file,
        tokenizer=tokenizer,
        num_frames=args.num_frames,
        image_size=args.image_size,
        prompt_len=args.prompt_len,
        choice_len=args.choice_len,
        prompt_variant=args.prompt_variant,
        teacher_dim=args.d_model,
        seed=args.seed + 1,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )

    model = CaTeRMini(
        d_model=args.d_model,
        vocab_size=args.vocab_size,
        image_size=args.image_size,
        max_frames=args.num_frames,
        clip_model_id=args.clip_model_id,
        debug=args.debug_model_prints,
        debug_forward_print_steps=args.debug_forward_print_steps,
    ).to(device)
    if args.debug_model_prints:
        print(
            "[debug:train:init] "
            f"model_params total={_param_count(model)} trainable={_param_count(model, trainable_only=True)}",
            flush=True,
        )
        print(
            "[debug:train:init] "
            f"clip_backbone total={_param_count(model.backbone)} trainable={_param_count(model.backbone, trainable_only=True)}",
            flush=True,
        )
        print(
            "[debug:train:init] "
            f"cross_attn trainable={_param_count(model.cross_attn, trainable_only=True)} "
            f"temporal_adapter trainable={_param_count(model.temporal_adapter, trainable_only=True)} "
            f"time_pos trainable={int(model.time_pos.numel()) if model.time_pos.requires_grad else 0}",
            flush=True,
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    total_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = max(1, total_steps // 20)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))).item()

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_acc = -1.0
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {"loss": 0.0, "ce_task": 0.0, "ce_teacher": 0.0, "mse": 0.0}

        for batch in train_loader:
            frames = batch["frames"].to(device)
            prompt_ids = batch["prompt_ids"].to(device)
            choice_ids = batch["choice_ids"].to(device)
            choice_mask = batch["choice_mask"].to(device)
            labels = batch["answer_idx"].to(device)
            teacher_feat = batch["teacher_feat"].to(device)
            has_teacher_feat = batch["has_teacher_feat"].to(device).bool()
            teacher_label = batch["teacher_label"].to(device)
            has_teacher_label = batch["has_teacher_label"].to(device).bool()

            optimizer.zero_grad(set_to_none=True)
            logits, video_feat = model(
                frames=frames,
                prompt_ids=prompt_ids,
                choice_ids=choice_ids,
                choice_mask=choice_mask,
            )

            ce_task = F.cross_entropy(logits, labels)
            ce_teacher = torch.zeros(1, device=device)
            mse = torch.zeros(1, device=device)

            if args.distill_mode in {"hard", "hybrid"} and has_teacher_label.any():
                ce_teacher = F.cross_entropy(logits[has_teacher_label], teacher_label[has_teacher_label])

            if args.distill_mode in {"soft", "hybrid"} and has_teacher_feat.any():
                if teacher_feat.size(-1) != video_feat.size(-1):
                    raise ValueError(
                        f"teacher dim ({teacher_feat.size(-1)}) != student dim ({video_feat.size(-1)}). "
                        "Please regenerate teacher_embedding with the same d_model."
                    )
                mse = F.mse_loss(video_feat[has_teacher_feat], teacher_feat[has_teacher_feat])

            loss = ce_task
            if args.distill_mode in {"hard", "hybrid"}:
                loss = loss + args.lambda_hard * ce_teacher
            if args.distill_mode in {"soft", "hybrid"}:
                loss = loss + args.lambda_soft * mse

            loss.backward()
            if args.debug_model_prints and global_step < args.debug_grad_print_steps:
                print(
                    "[debug:train:grad] "
                    f"step={global_step} "
                    f"cross_attn.in_proj_weight={_safe_grad_norm(model.cross_attn.in_proj_weight):.6f} "
                    f"time_pos={_safe_grad_norm(model.time_pos):.6f} "
                    f"temporal_adapter.layer0.self_attn.in_proj_weight="
                    f"{_safe_grad_norm(model.temporal_adapter.layers[0].self_attn.in_proj_weight):.6f}",
                    flush=True,
                )
            optimizer.step()
            scheduler.step()

            running["loss"] += loss.item()
            running["ce_task"] += ce_task.item()
            running["ce_teacher"] += ce_teacher.item()
            running["mse"] += mse.item()
            global_step += 1

        train_steps = max(1, len(train_loader))
        val_acc = evaluate(model, val_loader, device=device)
        lr = scheduler.get_last_lr()[0]
        print(
            f"epoch={epoch:02d} mode={args.distill_mode} prompt={args.prompt_variant} "
            f"loss={running['loss']/train_steps:.4f} ce_task={running['ce_task']/train_steps:.4f} "
            f"ce_teacher={running['ce_teacher']/train_steps:.4f} mse={running['mse']/train_steps:.4f} "
            f"val_acc={val_acc:.4f} lr={lr:.2e}",
            flush=True,
        )

        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint = {
                "model": model.state_dict(),
                "args": vars(args),
                "best_val_acc": best_acc,
                "global_step": global_step,
            }
            torch.save(checkpoint, out_dir / "best.pt")
            print(f"saved checkpoint to {out_dir / 'best.pt'}", flush=True)

    with (out_dir / "train_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    print(f"training done. best_val_acc={best_acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
