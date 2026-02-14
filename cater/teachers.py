import hashlib
import importlib
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image


def _stable_hash(text: str) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass
class TeacherOutput:
    embedding: torch.Tensor
    hard_idx: int


class BaseTeacher:
    def __init__(self, teacher_name: str, d_model: int = 512, seed: int = 42):
        self.teacher_name = teacher_name
        self.d_model = d_model
        self.seed = seed
        self._proj_cache: dict[int, torch.Tensor] = {}

    def infer(self, row: dict) -> TeacherOutput:
        raise NotImplementedError

    def _project_and_normalize(self, vec: torch.Tensor) -> torch.Tensor:
        vec = vec.detach().float().cpu()
        if vec.ndim != 1:
            vec = vec.flatten()
        in_dim = vec.numel()
        if in_dim == self.d_model:
            return F.normalize(vec, dim=0)

        if in_dim not in self._proj_cache:
            g = torch.Generator()
            g.manual_seed(self.seed + in_dim * 997 + (_stable_hash(self.teacher_name) % 100003))
            proj = torch.randn(self.d_model, in_dim, generator=g) / math.sqrt(in_dim)
            self._proj_cache[in_dim] = proj
        out = torch.matmul(self._proj_cache[in_dim], vec)
        return F.normalize(out, dim=0)


class VideoLLaVATeacher(BaseTeacher):
    def __init__(
        self,
        teacher_name: str,
        d_model: int = 512,
        seed: int = 42,
        model_id: str = "LanguageBind/Video-LLaVA-7B-hf",
        device: str = "auto",
        dtype: str = "auto",
        num_video_frames: int = 8,
        max_new_tokens: int = 8,
        load_in_4bit: bool = False,
    ):
        super().__init__(teacher_name=teacher_name, d_model=d_model, seed=seed)
        self.model_id = model_id
        self.num_video_frames = num_video_frames
        self.max_new_tokens = max_new_tokens

        self._torch_device = self._resolve_device(device)
        self._torch_dtype = self._resolve_dtype(dtype, self._torch_device)
        self.processor, self.model = self._load_model(load_in_4bit=load_in_4bit)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @staticmethod
    def _resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
        if dtype == "fp16":
            return torch.float16
        if dtype == "bf16":
            return torch.bfloat16
        if dtype == "fp32":
            return torch.float32
        if device.type == "cuda":
            return torch.float16
        return torch.float32

    def _load_model(self, load_in_4bit: bool):
        try:
            transformers = importlib.import_module("transformers")
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "transformers is required for Video-LLaVA real mode. "
                "Please install dependencies from requirements.txt."
            ) from e
        auto_processor = getattr(transformers, "AutoProcessor")
        processor = auto_processor.from_pretrained(self.model_id, trust_remote_code=True)

        model_kwargs = {
            "torch_dtype": self._torch_dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }

        if load_in_4bit:
            bnb_cfg_cls = getattr(transformers, "BitsAndBytesConfig", None)
            if bnb_cfg_cls is None:
                raise RuntimeError("bitsandbytes config is unavailable. Please install bitsandbytes first.")
            model_kwargs["quantization_config"] = bnb_cfg_cls(load_in_4bit=True)

        model = None
        # Try the native Video-LLaVA class first; this checkpoint is not a LlavaNextVideo checkpoint.
        for cls_name in [
            "VideoLlavaForConditionalGeneration",
            "LlavaNextVideoForConditionalGeneration",
            "LlavaOnevisionForConditionalGeneration",
            "AutoModelForVision2Seq",
        ]:
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                continue
            try:
                model = cls.from_pretrained(self.model_id, **model_kwargs)
                break
            except Exception:
                continue

        if model is None:
            raise RuntimeError(
                f"Failed to load Video-LLaVA model from {self.model_id}. "
                "Please verify model_id and transformers version."
            )

        if not load_in_4bit:
            try:
                model = model.to(self._torch_device)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to move model to device {self._torch_device}. "
                    "Check CUDA visibility or use --device cpu."
                ) from e
        model.eval()
        return processor, model

    @staticmethod
    def _uniform_indices(n_items: int, n_pick: int) -> list[int]:
        if n_items <= 0:
            return []
        if n_items <= n_pick:
            return list(range(n_items))
        if n_pick <= 1:
            return [0]
        return [round(i * (n_items - 1) / (n_pick - 1)) for i in range(n_pick)]

    def _load_frames(self, row: dict) -> list[Image.Image]:
        frame_paths = row.get("frame_paths", [])
        if not frame_paths:
            raise ValueError("Video-LLaVA real mode requires `frame_paths` in each row.")
        picked = self._uniform_indices(len(frame_paths), self.num_video_frames)
        frames: list[Image.Image] = []
        for idx in picked:
            path = Path(frame_paths[idx])
            with Image.open(path) as img:
                frames.append(img.convert("RGB"))
        return frames

    @staticmethod
    def _build_mcq_prompt(question: str, choices: list[str]) -> str:
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        choice_lines = [f"{letter}. {choice}" for letter, choice in zip(letters, choices)]
        return (
            "Answer the multiple-choice question based on the given video.\n"
            f"Question: {question}\n"
            "Choices:\n"
            + "\n".join(choice_lines)
            + "\nRespond with only the option letter."
        )

    def _build_letter_scoring_prompt(self, question: str, choices: list[str]) -> str:
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        choice_lines = [f"{letter}. {choice}" for letter, choice in zip(letters, choices)]
        return (
            "Answer the multiple-choice question based on the given video.\n"
            f"Question: {question}\n"
            "Choices:\n"
            + "\n".join(choice_lines)
            + "\nAnswer:"
        )

    def _encode_text_with_frames(self, text: str, frames: list[Image.Image], add_generation_prompt: bool) -> dict:
        if hasattr(self.processor, "apply_chat_template"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": text},
                    ],
                }
            ]
            try:
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            except Exception:
                pass

        # VideoLlavaProcessor expects a <video> placeholder token in text when videos are passed.
        # Some processor versions do not expose/accept chat templates, so enforce it here.
        if "<video>" not in text:
            text = f"<video>\n{text}"

        try:
            model_inputs = self.processor(
                text=text,
                videos=[frames],
                return_tensors="pt",
                padding=True,
            )
        except Exception:
            # Fallback for processors that expose only image interface.
            model_inputs = self.processor(
                text=text,
                images=frames,
                return_tensors="pt",
                padding=True,
            )

        out: dict = {}
        for key, value in model_inputs.items():
            if hasattr(value, "to"):
                out[key] = value.to(self._torch_device)
            else:
                out[key] = value
        return out

    def _prepare_inputs(self, question: str, choices: list[str], frames: list[Image.Image]) -> dict:
        prompt = self._build_mcq_prompt(question=question, choices=choices)
        return self._encode_text_with_frames(text=prompt, frames=frames, add_generation_prompt=True)

    def _score_choice_letters(self, question: str, choices: list[str], frames: list[Image.Image], fallback_idx: int) -> int:
        if not choices:
            return 0
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        base_prompt = self._build_letter_scoring_prompt(question=question, choices=choices)
        base_inputs = self._encode_text_with_frames(
            text=base_prompt,
            frames=frames,
            add_generation_prompt=False,
        )
        base_ids = base_inputs.get("input_ids")
        if base_ids is None or base_ids.ndim != 2:
            return fallback_idx
        base_len = int(base_ids.shape[1])

        scores: list[float] = []
        for letter in letters:
            full_text = f"{base_prompt} {letter}"
            full_inputs = self._encode_text_with_frames(
                text=full_text,
                frames=frames,
                add_generation_prompt=False,
            )
            input_ids = full_inputs.get("input_ids")
            if input_ids is None or input_ids.ndim != 2:
                scores.append(float("-inf"))
                continue
            seq_len = int(input_ids.shape[1])
            if seq_len <= base_len:
                scores.append(float("-inf"))
                continue

            with torch.inference_mode():
                outputs = self.model(**full_inputs, return_dict=True)
            logits = outputs.logits
            if logits is None or logits.ndim != 3 or logits.size(1) != seq_len:
                scores.append(float("-inf"))
                continue

            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            targets = input_ids[:, 1:]
            token_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [1, seq_len-1]

            # Token at original position p is predicted by logit at p-1.
            start_idx = max(base_len - 1, 0)
            ans_logp = token_logp[:, start_idx:]

            attn = full_inputs.get("attention_mask")
            if attn is not None and attn.ndim == 2 and attn.size(1) == seq_len:
                ans_mask = attn[:, 1:][:, start_idx:].float()
                denom = ans_mask.sum(dim=1).clamp(min=1.0)
                score = ((ans_logp * ans_mask).sum(dim=1) / denom)[0].item()
            else:
                score = ans_logp.mean(dim=1)[0].item()
            scores.append(score)

        if not scores:
            return fallback_idx
        best = int(torch.tensor(scores).argmax().item())
        if 0 <= best < len(choices):
            return best
        return fallback_idx

    def _extract_soft_embedding(self, inputs: dict) -> torch.Tensor:
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)

        hidden = getattr(outputs, "hidden_states", None)
        if hidden is not None and len(hidden) > 0:
            last = hidden[-1]
            attn = inputs.get("attention_mask")
            if attn is not None and attn.ndim == 2 and attn.size(1) == last.size(1):
                w = attn.float().unsqueeze(-1)
                vec = (last * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
            else:
                vec = last.mean(dim=1)
            return self._project_and_normalize(vec[0])

        logits = getattr(outputs, "logits", None)
        if logits is not None:
            return self._project_and_normalize(logits[0].float().mean(dim=0))

        raise RuntimeError("Video-LLaVA output does not provide hidden_states/logits for soft embedding extraction.")

    def infer(self, row: dict) -> TeacherOutput:
        question = row.get("question", "")
        choices = row.get("choices", [])
        answer_idx = int(row.get("answer_idx", 0))
        frames = self._load_frames(row)
        hard_idx = self._score_choice_letters(
            question=question,
            choices=choices,
            frames=frames,
            fallback_idx=answer_idx,
        )
        inputs = self._prepare_inputs(question=question, choices=choices, frames=frames)
        embedding = self._extract_soft_embedding(inputs)
        return TeacherOutput(embedding=embedding, hard_idx=hard_idx)


class FlamingoTeacher(BaseTeacher):
    def infer(self, row: dict) -> TeacherOutput:
        raise NotImplementedError(
            "Real Flamingo inference is not wired in this repo yet. "
            "Please connect OpenFlamingo checkpoints/inference pipeline."
        )


class ChatUniVLTeacher(BaseTeacher):
    def infer(self, row: dict) -> TeacherOutput:
        raise NotImplementedError(
            "Real Chat-UniVL inference is not wired in this repo yet. "
            "Please connect your Chat-UniVL inference pipeline."
        )


def build_teacher(
    teacher_name: str,
    d_model: int,
    seed: int,
    model_id: str = "LanguageBind/Video-LLaVA-7B-hf",
    device: str = "auto",
    dtype: str = "auto",
    num_video_frames: int = 8,
    max_new_tokens: int = 8,
    load_in_4bit: bool = False,
) -> BaseTeacher:
    teacher_name = teacher_name.lower()

    if teacher_name == "videollava":
        return VideoLLaVATeacher(
            teacher_name=teacher_name,
            d_model=d_model,
            seed=seed,
            model_id=model_id,
            device=device,
            dtype=dtype,
            num_video_frames=num_video_frames,
            max_new_tokens=max_new_tokens,
            load_in_4bit=load_in_4bit,
        )
    if teacher_name == "flamingo":
        return FlamingoTeacher(teacher_name=teacher_name, d_model=d_model, seed=seed)
    if teacher_name == "chatunivl":
        return ChatUniVLTeacher(teacher_name=teacher_name, d_model=d_model, seed=seed)
    raise ValueError(f"Unknown teacher_name: {teacher_name}")
