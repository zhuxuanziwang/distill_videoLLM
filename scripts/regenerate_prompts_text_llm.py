#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

try:
    import torch
    from transformers import pipeline
except ModuleNotFoundError as e:  # pragma: no cover
    raise RuntimeError("transformers/torch is required. Please install requirements.txt first.") from e

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x


JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")


def parse_prompt_list(text: str, expected_n: int) -> list[str]:
    raw = text.strip()
    arr = None
    try:
        arr = json.loads(raw)
    except Exception:
        m = JSON_ARRAY_RE.search(raw)
        if m:
            arr = json.loads(m.group(0))

    if not isinstance(arr, list):
        raise ValueError(f"Output is not a JSON list. text={raw[:240]}")

    prompts: list[str] = []
    for x in arr:
        if isinstance(x, str):
            s = " ".join(x.strip().split())
            if s:
                prompts.append(s.rstrip(".") + ".")
    if not prompts:
        raise ValueError("Empty prompts after parsing.")
    if len(prompts) < expected_n:
        prompts.extend([prompts[-1]] * (expected_n - len(prompts)))
    return prompts[:expected_n]


def build_instruction(row: dict, source_prompts: list[str], expected_n: int) -> str:
    question = row.get("question", "").strip()
    choices = row.get("choices", [])
    choice_text = "\n".join([f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices)])
    notes = "\n".join([f"- Segment {i + 1}: {p}" for i, p in enumerate(source_prompts[:expected_n])])
    return (
        "You are refining event-level prompts for video question answering.\n"
        f"Question: {question}\n"
        f"Choices:\n{choice_text}\n\n"
        "Visual notes extracted from the corresponding video segments:\n"
        f"{notes}\n\n"
        f"Write exactly {expected_n} concise factual event prompts (one per segment), in temporal order.\n"
        "Return ONLY a JSON array of strings. No markdown."
    )


def strip_prompt_from_generation(prompt: str, generated_text: str) -> str:
    if generated_text.startswith(prompt):
        return generated_text[len(prompt):].strip()
    return generated_text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate event prompts with a text-only LLM for generator ablation. "
            "This script keeps visual evidence fixed via existing prompt notes."
        )
    )
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True, help="e.g., lmsys/vicuna-7b-v1.5")
    parser.add_argument("--source_prompt_field", type=str, default="event_prompts")
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--torch_dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--log_every", type=int, default=20)
    args = parser.parse_args()

    dtype = None
    if args.torch_dtype == "fp16":
        dtype = torch.float16
    elif args.torch_dtype == "bf16":
        dtype = torch.bfloat16
    elif args.torch_dtype == "fp32":
        dtype = torch.float32

    text_gen = pipeline(
        task="text-generation",
        model=args.model_id,
        tokenizer=args.model_id,
        device_map="auto",
        torch_dtype=dtype,
    )

    in_path = Path(args.input_jsonl)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    kept = 0
    failed = 0
    with out_path.open("w", encoding="utf-8") as wf:
        for i, row in enumerate(tqdm(rows, desc="regen_prompts"), start=1):
            source_prompts = row.get(args.source_prompt_field, [])
            if not isinstance(source_prompts, list) or len(source_prompts) == 0:
                failed += 1
                continue
            expected_n = len(source_prompts)
            prompt = build_instruction(row=row, source_prompts=source_prompts, expected_n=expected_n)
            try:
                out = text_gen(
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    return_full_text=True,
                )
                generated = out[0]["generated_text"]
                answer_text = strip_prompt_from_generation(prompt, generated)
                event_prompts = parse_prompt_list(answer_text, expected_n=expected_n)
            except Exception:
                # Fallback: keep original source prompts if generation/parsing fails.
                event_prompts = [str(x).strip() for x in source_prompts[:expected_n]]
                if len(event_prompts) < expected_n:
                    event_prompts.extend([event_prompts[-1]] * (expected_n - len(event_prompts)))

            row["event_prompts"] = event_prompts
            row["frame_prompts"] = event_prompts.copy()
            row["prompt_generator"] = args.model_id
            wf.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

            if args.log_every > 0 and i % args.log_every == 0:
                wf.flush()
                print(f"processed={i}/{len(rows)} kept={kept} failed={failed}", flush=True)

    print(f"done. wrote={out_path} kept={kept} failed={failed}")


if __name__ == "__main__":
    main()
