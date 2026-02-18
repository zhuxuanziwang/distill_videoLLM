import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(3, 1, 1)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(3, 1, 1)


def _normalize_clip(x: torch.Tensor) -> torch.Tensor:
    return (x - _CLIP_MEAN) / _CLIP_STD


def _render_frame(action_id: int, t: int, size: int, seed: int) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed + t)
    frame = torch.rand((3, size, size), generator=g) * 0.08

    base = torch.tensor(
        [
            ((action_id * 7) % 11) / 10.0,
            ((action_id * 5 + 3) % 11) / 10.0,
            ((action_id * 3 + 1) % 11) / 10.0,
        ],
        dtype=torch.float32,
    )
    frame = frame + base.view(3, 1, 1) * 0.45

    block = max(8, size // 6)
    span = max(1, size - block)
    x = (action_id * 17 + t * 19) % span
    y = (action_id * 13 + t * 11) % span
    channel = action_id % 3
    frame[channel, y : y + block, x : x + block] += 0.4

    return _normalize_clip(frame.clamp(0.0, 1.0))


class ToyVideoQADataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer,
        num_frames: int = 6,
        image_size: int = 96,
        prompt_len: int = 32,
        choice_len: int = 48,
    ):
        self.path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.num_frames = num_frames
        self.image_size = image_size
        self.prompt_len = prompt_len
        self.choice_len = choice_len

        with self.path.open("r", encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        action_id = int(row["action_id"])
        seed = int(row["frame_seed"])
        question = row["question"]
        choices = row["choices"]
        prompts = row["event_prompts"]

        if len(prompts) < self.num_frames:
            prompts = prompts + [prompts[-1]] * (self.num_frames - len(prompts))
        prompts = prompts[: self.num_frames]

        frames = torch.stack(
            [_render_frame(action_id=action_id, t=t, size=self.image_size, seed=seed) for t in range(self.num_frames)]
        )
        prompt_ids = torch.stack([self.tokenizer.encode(text, self.prompt_len) for text in prompts])
        choice_ids = torch.stack(
            [self.tokenizer.encode(f"{question} answer: {choice}", self.choice_len) for choice in choices]
        )

        return {
            "frames": frames,
            "prompt_ids": prompt_ids,
            "choice_ids": choice_ids,
            "choice_mask": torch.ones(choice_ids.size(0), dtype=torch.bool),
            "answer_idx": torch.tensor(int(row["answer_idx"]), dtype=torch.long),
            "teacher_feat": torch.tensor(row["teacher_embedding"], dtype=torch.float32),
            "has_teacher_feat": torch.tensor(1, dtype=torch.long),
            "teacher_label": torch.tensor(int(row.get("teacher_hard_idx", -1)), dtype=torch.long),
            "has_teacher_label": torch.tensor(int("teacher_hard_idx" in row), dtype=torch.long),
        }


def collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_choices = max(sample["choice_ids"].size(0) for sample in batch)
    choice_len = batch[0]["choice_ids"].size(1)

    collated: dict[str, torch.Tensor] = {}
    for key in ["frames", "prompt_ids", "answer_idx", "teacher_feat", "has_teacher_feat", "teacher_label", "has_teacher_label"]:
        collated[key] = torch.stack([sample[key] for sample in batch], dim=0)

    padded_choices = []
    choice_mask = []
    for sample in batch:
        c = sample["choice_ids"].size(0)
        if c < max_choices:
            pad = torch.zeros(max_choices - c, choice_len, dtype=sample["choice_ids"].dtype)
            padded = torch.cat([sample["choice_ids"], pad], dim=0)
        else:
            padded = sample["choice_ids"]
        padded_choices.append(padded)
        mask = torch.zeros(max_choices, dtype=torch.bool)
        mask[:c] = True
        choice_mask.append(mask)

    collated["choice_ids"] = torch.stack(padded_choices, dim=0)
    collated["choice_mask"] = torch.stack(choice_mask, dim=0)
    return collated


def _load_frame_from_path(path: str, image_size: int) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        x = torch.tensor(list(img.getdata()), dtype=torch.float32).view(image_size, image_size, 3)
        x = x.permute(2, 0, 1) / 255.0
    return _normalize_clip(x)


class VideoQADataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer,
        num_frames: int = 6,
        image_size: int = 224,
        prompt_len: int = 32,
        choice_len: int = 64,
        prompt_variant: str = "event",
        teacher_dim: int = 512,
        seed: int = 42,
    ):
        self.path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.num_frames = num_frames
        self.image_size = image_size
        self.prompt_len = prompt_len
        self.choice_len = choice_len
        self.prompt_variant = prompt_variant
        self.teacher_dim = teacher_dim
        self.rng = random.Random(seed)

        with self.path.open("r", encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f if line.strip()]

        self.prompt_bank = []
        for row in self.rows:
            self.prompt_bank.extend(row.get("event_prompts", []))
            self.prompt_bank.extend(row.get("frame_prompts", []))
        if not self.prompt_bank:
            self.prompt_bank = ["A person is doing an action."]

    def __len__(self) -> int:
        return len(self.rows)

    def _select_prompts(self, row: dict, idx: int) -> list[str]:
        event_prompts = row.get("event_prompts", [])
        frame_prompts = row.get("frame_prompts", [])
        prompts = event_prompts
        if self.prompt_variant == "frame":
            prompts = frame_prompts if frame_prompts else event_prompts
        elif self.prompt_variant == "none":
            prompts = [""]
        elif self.prompt_variant == "random":
            local_rng = random.Random(idx + 1337)
            prompts = [local_rng.choice(self.prompt_bank) for _ in range(self.num_frames)]

        if not prompts:
            prompts = [""]
        if len(prompts) < self.num_frames:
            prompts = prompts + [prompts[-1]] * (self.num_frames - len(prompts))
        return prompts[: self.num_frames]

    def _build_frames(self, row: dict) -> torch.Tensor:
        frame_paths = row.get("frame_paths")
        if frame_paths:
            if len(frame_paths) < self.num_frames:
                frame_paths = frame_paths + [frame_paths[-1]] * (self.num_frames - len(frame_paths))
            frame_paths = frame_paths[: self.num_frames]
            return torch.stack([_load_frame_from_path(path, self.image_size) for path in frame_paths], dim=0)

        action_id = int(row.get("action_id", 0))
        seed = int(row.get("frame_seed", 42))
        return torch.stack(
            [_render_frame(action_id=action_id, t=t, size=self.image_size, seed=seed) for t in range(self.num_frames)],
            dim=0,
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        question = row["question"]
        choices = row["choices"]
        prompts = self._select_prompts(row, idx)

        frames = self._build_frames(row)
        prompt_ids = torch.stack([self.tokenizer.encode(text, self.prompt_len) for text in prompts], dim=0)
        choice_ids = torch.stack(
            [self.tokenizer.encode(f"{question} answer: {choice}", self.choice_len) for choice in choices],
            dim=0,
        )

        teacher_embedding = row.get("teacher_embedding")
        if teacher_embedding is None:
            teacher_feat = torch.zeros(self.teacher_dim, dtype=torch.float32)
            has_teacher_feat = torch.tensor(0, dtype=torch.long)
        else:
            teacher_feat = torch.tensor(teacher_embedding, dtype=torch.float32)
            has_teacher_feat = torch.tensor(1, dtype=torch.long)

        teacher_label = row.get("teacher_hard_idx", -1)
        has_teacher_label = torch.tensor(int(teacher_label >= 0), dtype=torch.long)
        teacher_label_t = torch.tensor(max(-1, int(teacher_label)), dtype=torch.long)

        return {
            "frames": frames,
            "prompt_ids": prompt_ids,
            "choice_ids": choice_ids,
            "choice_mask": torch.ones(choice_ids.size(0), dtype=torch.bool),
            "answer_idx": torch.tensor(int(row["answer_idx"]), dtype=torch.long),
            "teacher_feat": teacher_feat,
            "has_teacher_feat": has_teacher_feat,
            "teacher_label": teacher_label_t,
            "has_teacher_label": has_teacher_label,
        }
