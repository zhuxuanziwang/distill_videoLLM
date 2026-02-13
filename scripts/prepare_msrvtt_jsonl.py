#!/usr/bin/env python3
import argparse
import base64
import io
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

from PIL import Image

try:
    from openai import OpenAI
except ModuleNotFoundError as e:  # pragma: no cover
    raise RuntimeError("openai package is required. Please install requirements.txt first.") from e

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x


JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")


def normalize_answer(text: str) -> str:
    return " ".join(text.strip().lower().split())


def uniform_indices(n: int, k: int) -> list[int]:
    if n <= 0:
        return []
    if n <= k:
        return list(range(n))
    if k <= 1:
        return [0]
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]


def build_answer_pool(train_rows: list[dict], min_freq: int) -> list[str]:
    cnt = Counter(normalize_answer(r["answer"]) for r in train_rows)
    pool = [ans for ans, freq in cnt.items() if freq >= min_freq]
    if not pool:
        pool = list(cnt.keys())
    return sorted(pool)


def encode_image_data_url(path: Path, max_side: int, jpeg_quality: int) -> str:
    with Image.open(path) as img:
        img = img.convert("RGB")
        w, h = img.size
        long_side = max(w, h)
        if long_side > max_side:
            scale = max_side / float(long_side)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            img = img.resize((nw, nh), Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def parse_prompt_list(text: str, expected_n: int) -> list[str]:
    raw = text.strip()
    arr = None
    try:
        arr = json.loads(raw)
    except Exception:
        match = JSON_ARRAY_RE.search(raw)
        if match:
            arr = json.loads(match.group(0))
    if not isinstance(arr, list):
        raise ValueError(f"model output is not JSON array: {raw[:300]}")

    clean = []
    for item in arr:
        if isinstance(item, str):
            s = " ".join(item.strip().split())
            if s:
                clean.append(s.rstrip(".") + ".")
    if not clean:
        raise ValueError("empty prompt list")
    if len(clean) < expected_n:
        clean.extend([clean[-1]] * (expected_n - len(clean)))
    return clean[:expected_n]


def call_openai_for_prompts(
    client: OpenAI,
    model: str,
    question: str,
    choices: list[str],
    frame_paths: list[str],
    image_detail: str,
    max_side: int,
    jpeg_quality: int,
    max_retries: int,
    timeout_sec: float,
) -> list[str]:
    n = len(frame_paths)
    choice_text = "\n".join([f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices)])
    instruction = (
        "You are generating event-level prompts for video understanding.\n"
        f"You are given {n} temporal frames in order.\n"
        "Return ONLY a JSON array with one concise factual sentence per frame.\n"
        "No markdown, no extra keys.\n\n"
        f"Question: {question}\n"
        f"Choices:\n{choice_text}\n"
    )
    content = [{"type": "input_text", "text": instruction}]
    for fp in frame_paths:
        content.append(
            {
                "type": "input_image",
                "image_url": encode_image_data_url(Path(fp), max_side=max_side, jpeg_quality=jpeg_quality),
                "detail": image_detail,
            }
        )
    content.append({"type": "input_text", "text": f"Output exactly {n} sentences as a JSON array."})

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.responses.create(
                model=model,
                input=[{"role": "user", "content": content}],
                temperature=0.0,
                timeout=timeout_sec,
            )
            return parse_prompt_list(resp.output_text, expected_n=n)
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"OpenAI prompt generation failed after retries: {last_err}")


def sample_choices(correct: str, pool: list[str], n_choices: int, rng: random.Random) -> tuple[list[str], int]:
    negatives = [x for x in pool if x != correct]
    if len(negatives) < n_choices - 1:
        raise RuntimeError("Answer pool is too small for requested number of choices.")
    wrong = rng.sample(negatives, n_choices - 1)
    choices = [correct] + wrong
    rng.shuffle(choices)
    return choices, choices.index(correct)


def convert_split(
    split_name: str,
    qa_rows: list[dict],
    frames_root: Path,
    out_path: Path,
    answer_pool: list[str],
    n_choices: int,
    n_frames: int,
    seed: int,
    max_samples: int,
    allow_missing_frames: bool,
    client: OpenAI,
    openai_model: str,
    image_detail: str,
    max_side: int,
    jpeg_quality: int,
    openai_max_retries: int,
    openai_sleep_sec: float,
    openai_timeout_sec: float,
    log_every: int,
) -> tuple[int, int]:
    rng = random.Random(seed)
    rows = qa_rows
    if max_samples > 0:
        rows = rows[:max_samples]

    kept = 0
    dropped = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as wf:
        for idx, row in enumerate(tqdm(rows, desc=f"convert_{split_name}", total=len(rows)), start=1):
            vid = f"video{int(row['video_id'])}"
            frame_dir = frames_root / vid
            frame_files = sorted(frame_dir.glob("frame_*.jpg"))
            if len(frame_files) == 0 and not allow_missing_frames:
                dropped += 1
                continue
            if len(frame_files) == 0:
                # Keep row only when explicitly allowed, prompts will be empty.
                sampled_paths = []
            else:
                picks = uniform_indices(len(frame_files), n_frames)
                sampled_paths = [str(frame_files[i]) for i in picks]

            correct = normalize_answer(row["answer"])
            choices, answer_idx = sample_choices(correct, answer_pool, n_choices=n_choices, rng=rng)
            question = row["question"].strip()
            if sampled_paths:
                event_prompts = call_openai_for_prompts(
                    client=client,
                    model=openai_model,
                    question=question,
                    choices=choices,
                    frame_paths=sampled_paths,
                    image_detail=image_detail,
                    max_side=max_side,
                    jpeg_quality=jpeg_quality,
                    max_retries=openai_max_retries,
                    timeout_sec=openai_timeout_sec,
                )
            else:
                event_prompts = ["" for _ in range(n_frames)]
            frame_prompts = event_prompts.copy()
            if openai_sleep_sec > 0:
                time.sleep(openai_sleep_sec)

            out = {
                "id": f"msrvtt_{split_name}_{row['id']}",
                "dataset": "msrvtt_qa",
                "source_id": int(row["id"]),
                "video_id": int(row["video_id"]),
                "category_id": int(row["category_id"]),
                "question": question,
                "choices": choices,
                "answer_idx": answer_idx,
                "event_prompts": event_prompts,
                "frame_prompts": frame_prompts,
                "frame_paths": sampled_paths,
            }
            wf.write(json.dumps(out, ensure_ascii=False) + "\n")
            kept += 1
            if log_every > 0 and idx % log_every == 0:
                wf.flush()
                print(
                    f"[{split_name}] processed={idx}/{len(rows)} kept={kept} dropped={dropped}",
                    flush=True,
                )

    return kept, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MSRVTT-QA raw json to project jsonl format.")
    parser.add_argument("--qa_dir", type=str, default="data/msrvtt_raw/msrvtt_qa/MSRVTT-QA")
    parser.add_argument("--frames_root", type=str, default="data/msrvtt_frames")
    parser.add_argument("--out_dir", type=str, default="data/msrvtt_full_jsonl")
    parser.add_argument("--num_choices", type=int, default=5)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--min_answer_freq", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train", type=int, default=0)
    parser.add_argument("--max_val", type=int, default=0)
    parser.add_argument("--max_test", type=int, default=0)
    parser.add_argument("--allow_missing_frames", action="store_true")
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--openai_api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--openai_image_detail", type=str, default="low", choices=["low", "high", "auto"])
    parser.add_argument("--openai_max_side", type=int, default=512)
    parser.add_argument("--openai_jpeg_quality", type=int, default=75)
    parser.add_argument("--openai_max_retries", type=int, default=5)
    parser.add_argument("--openai_sleep_sec", type=float, default=0.0)
    parser.add_argument("--openai_timeout_sec", type=float, default=120.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated subset of splits to process, e.g. 'val,test'.",
    )
    args = parser.parse_args()

    api_key = os.environ.get(args.openai_api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.openai_api_key_env} is not set.")
    client = OpenAI(api_key=api_key)

    qa_dir = Path(args.qa_dir)
    frames_root = Path(args.frames_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_set = {s.strip() for s in args.splits.split(",") if s.strip()}
    valid_splits = {"train", "val", "test"}
    invalid = split_set - valid_splits
    if invalid:
        raise ValueError(f"Invalid split(s): {sorted(invalid)}. Valid: {sorted(valid_splits)}")

    train_rows = json.load(open(qa_dir / "train_qa.json", "r", encoding="utf-8"))
    val_rows = json.load(open(qa_dir / "val_qa.json", "r", encoding="utf-8"))
    test_rows = json.load(open(qa_dir / "test_qa.json", "r", encoding="utf-8"))

    answer_pool = build_answer_pool(train_rows, min_freq=args.min_answer_freq)
    print(f"answer_pool_size={len(answer_pool)} min_freq={args.min_answer_freq}")

    tr_kept = tr_drop = va_kept = va_drop = te_kept = te_drop = 0

    if "train" in split_set:
        tr_kept, tr_drop = convert_split(
            split_name="train",
            qa_rows=train_rows,
            frames_root=frames_root,
            out_path=out_dir / "train.jsonl",
            answer_pool=answer_pool,
            n_choices=args.num_choices,
            n_frames=args.num_frames,
            seed=args.seed,
            max_samples=args.max_train,
            allow_missing_frames=args.allow_missing_frames,
            client=client,
            openai_model=args.openai_model,
            image_detail=args.openai_image_detail,
            max_side=args.openai_max_side,
            jpeg_quality=args.openai_jpeg_quality,
            openai_max_retries=args.openai_max_retries,
            openai_sleep_sec=args.openai_sleep_sec,
            openai_timeout_sec=args.openai_timeout_sec,
            log_every=args.log_every,
        )
    else:
        print("skip train split", flush=True)

    if "val" in split_set:
        va_kept, va_drop = convert_split(
            split_name="val",
            qa_rows=val_rows,
            frames_root=frames_root,
            out_path=out_dir / "val.jsonl",
            answer_pool=answer_pool,
            n_choices=args.num_choices,
            n_frames=args.num_frames,
            seed=args.seed + 1,
            max_samples=args.max_val,
            allow_missing_frames=args.allow_missing_frames,
            client=client,
            openai_model=args.openai_model,
            image_detail=args.openai_image_detail,
            max_side=args.openai_max_side,
            jpeg_quality=args.openai_jpeg_quality,
            openai_max_retries=args.openai_max_retries,
            openai_sleep_sec=args.openai_sleep_sec,
            openai_timeout_sec=args.openai_timeout_sec,
            log_every=args.log_every,
        )
    else:
        print("skip val split", flush=True)

    if "test" in split_set:
        te_kept, te_drop = convert_split(
            split_name="test",
            qa_rows=test_rows,
            frames_root=frames_root,
            out_path=out_dir / "test.jsonl",
            answer_pool=answer_pool,
            n_choices=args.num_choices,
            n_frames=args.num_frames,
            seed=args.seed + 2,
            max_samples=args.max_test,
            allow_missing_frames=args.allow_missing_frames,
            client=client,
            openai_model=args.openai_model,
            image_detail=args.openai_image_detail,
            max_side=args.openai_max_side,
            jpeg_quality=args.openai_jpeg_quality,
            openai_max_retries=args.openai_max_retries,
            openai_sleep_sec=args.openai_sleep_sec,
            openai_timeout_sec=args.openai_timeout_sec,
            log_every=args.log_every,
        )
    else:
        print("skip test split", flush=True)

    print(
        f"done. out_dir={out_dir} "
        f"train={tr_kept}(drop={tr_drop}) val={va_kept}(drop={va_drop}) test={te_kept}(drop={te_drop})"
    )


if __name__ == "__main__":
    main()
