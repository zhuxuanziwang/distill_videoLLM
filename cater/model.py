import importlib

import torch
import torch.nn.functional as F
from torch import nn


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.float().unsqueeze(-1)
    denom = weights.sum(dim=1).clamp(min=1.0)
    return (x * weights).sum(dim=1) / denom


class FrozenDualEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        model_id: str = "openai/clip-vit-base-patch16",
    ):
        super().__init__()
        self.d_model = d_model
        self.model_id = model_id
        self.clip = self._load_clip(model_id=model_id)
        self.pad_token_id = int(getattr(self.clip.config.text_config, "pad_token_id", 1))
        proj_dim = int(getattr(self.clip.config, "projection_dim", d_model))
        if d_model != proj_dim:
            raise ValueError(
                f"d_model ({d_model}) must match CLIP projection dim ({proj_dim}) for this implementation."
            )

        for param in self.clip.parameters():
            param.requires_grad = False
        self.eval()

    def train(self, mode: bool = True):
        # Keep frozen CLIP in eval mode even when outer module switches to train().
        super().train(False)
        self.clip.eval()
        return self

    @staticmethod
    def _load_clip(model_id: str):
        try:
            transformers = importlib.import_module("transformers")
        except ModuleNotFoundError as e:
            raise RuntimeError("transformers is required for CLIP backbone. Please install requirements.txt.") from e
        clip_cls = getattr(transformers, "CLIPModel")
        return clip_cls.from_pretrained(model_id)

    def encode_visual(self, frames: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = frames.shape
        pixel_values = frames.view(b * t, c, h, w)
        vision_out = self.clip.vision_model(pixel_values=pixel_values, return_dict=True)
        tokens = vision_out.last_hidden_state  # [b*t, n_tokens, vision_hidden]
        proj_w = self.clip.visual_projection.weight  # [d_model, vision_hidden]
        tokens = torch.matmul(tokens, proj_w.t())
        return tokens.view(b, t, tokens.size(1), tokens.size(2))

    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        attention_mask = token_ids.ne(self.pad_token_id).long()
        text_out = self.clip.text_model(
            input_ids=token_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        tokens = text_out.last_hidden_state  # [b, l, text_hidden]
        proj_w = self.clip.text_projection.weight  # [d_model, text_hidden]
        return torch.matmul(tokens, proj_w.t())


class CaTeRMini(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        vocab_size: int = 8192,
        image_size: int = 224,
        patch_size: int = 16,
        max_text_len: int = 77,
        max_frames: int = 8,
        dropout: float = 0.1,
        clip_model_id: str = "openai/clip-vit-base-patch16",
        debug: bool = False,
        debug_forward_print_steps: int = 1,
    ):
        super().__init__()
        self.debug = debug
        self.debug_forward_print_steps = max(0, int(debug_forward_print_steps))
        self._debug_forward_calls = 0

        self.backbone = FrozenDualEncoder(
            d_model=d_model,
            model_id=clip_model_id,
        )

        self.q_ln = nn.LayerNorm(d_model)
        self.kv_ln = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion_drop = nn.Dropout(dropout)
        self.fusion_ln = nn.LayerNorm(d_model)

        self.time_pos = nn.Parameter(torch.randn(1, max_frames, d_model) * 0.02)
        t_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_adapter = nn.TransformerEncoder(t_layer, num_layers=2)
        self.temporal_ln = nn.LayerNorm(d_model)

        if self.debug:
            self._print_debug_arch_summary(d_model=d_model, max_frames=max_frames)

    def _print_debug_arch_summary(self, d_model: int, max_frames: int) -> None:
        backbone_total = sum(p.numel() for p in self.backbone.parameters())
        backbone_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        cross_attn_params = sum(p.numel() for p in self.cross_attn.parameters())
        adapter_params = sum(p.numel() for p in self.temporal_adapter.parameters())
        print(
            "[debug:model:init] "
            f"d_model={d_model} max_frames={max_frames} "
            f"cross_attn(num_heads={self.cross_attn.num_heads}, dropout={self.cross_attn.dropout})",
            flush=True,
        )
        print(
            "[debug:model:init] "
            f"frozen_clip_params total={backbone_total} trainable={backbone_trainable}",
            flush=True,
        )
        print(
            "[debug:model:init] "
            f"time_pos shape={tuple(self.time_pos.shape)} init_std=0.02 requires_grad={self.time_pos.requires_grad}",
            flush=True,
        )
        print(
            "[debug:model:init] "
            f"temporal_adapter layers={len(self.temporal_adapter.layers)} params={adapter_params} "
            f"cross_attn_params={cross_attn_params}",
            flush=True,
        )

    def forward(
        self,
        frames: torch.Tensor,
        prompt_ids: torch.Tensor,
        choice_ids: torch.Tensor,
        choice_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, _, _, _ = frames.shape
        c = choice_ids.size(1)
        prompt_len = prompt_ids.size(-1)
        choice_len = choice_ids.size(-1)

        with torch.no_grad():
            visual_tokens = self.backbone.encode_visual(frames)
            prompt_tokens = self.backbone.encode_text(prompt_ids.view(b * t, prompt_len))
            choice_tokens = self.backbone.encode_text(choice_ids.view(b * c, choice_len))

        n, d = visual_tokens.size(2), visual_tokens.size(3)
        l_prompt = prompt_tokens.size(1)

        visual_bt = visual_tokens.view(b * t, n, d)
        prompt_bt = prompt_tokens.view(b * t, l_prompt, d)
        if self.debug and self._debug_forward_calls < self.debug_forward_print_steps:
            print(
                "[debug:model:fwd] "
                f"frames={tuple(frames.shape)} visual_tokens={tuple(visual_tokens.shape)} "
                f"prompt_tokens={tuple(prompt_tokens.shape)} choice_tokens={tuple(choice_tokens.shape)}",
                flush=True,
            )
            print(
                "[debug:model:fwd] "
                f"cross_attn query={tuple(visual_bt.shape)} key/value={tuple(prompt_bt.shape)}",
                flush=True,
            )
        fused_bt, _ = self.cross_attn(
            query=self.q_ln(visual_bt),
            key=self.kv_ln(prompt_bt),
            value=self.kv_ln(prompt_bt),
            need_weights=False,
        )
        fused_bt = self.fusion_ln(self.fusion_drop(fused_bt) + visual_bt)
        frame_vec = fused_bt.mean(dim=1).view(b, t, d)

        frame_vec = frame_vec + self.time_pos[:, :t]
        temporal = self.temporal_adapter(frame_vec)
        temporal = self.temporal_ln(temporal)
        h_video = temporal.mean(dim=1)

        choice_token_mask = choice_ids.view(b * c, choice_len).ne(0)
        choice_vec = _masked_mean(choice_tokens, choice_token_mask).view(b, c, d)

        h_video = F.normalize(h_video, dim=-1)
        choice_vec = F.normalize(choice_vec, dim=-1)
        logits = torch.einsum("bd,bcd->bc", h_video, choice_vec)
        if choice_mask is not None:
            logits = logits.masked_fill(~choice_mask, -1e9)
        if self.debug and self._debug_forward_calls < self.debug_forward_print_steps:
            print(
                "[debug:model:fwd] "
                f"frame_vec={tuple(frame_vec.shape)} h_video={tuple(h_video.shape)} logits={tuple(logits.shape)}",
                flush=True,
            )
        self._debug_forward_calls += 1
        return logits, h_video
