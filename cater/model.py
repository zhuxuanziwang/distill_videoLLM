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
        vocab_size: int = 8192,
        max_text_len: int = 77,
        patch_size: int = 16,
        image_size: int = 96,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_text_len = max_text_len
        self.patch_size = patch_size
        self.image_size = image_size

        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        n_patches = (image_size // patch_size) ** 2
        self.visual_pos = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        vis_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.visual_encoder = nn.TransformerEncoder(vis_layer, num_layers=1)
        self.visual_ln = nn.LayerNorm(d_model)

        self.text_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.text_pos = nn.Parameter(torch.randn(1, max_text_len, d_model) * 0.02)
        txt_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.text_encoder = nn.TransformerEncoder(txt_layer, num_layers=2)
        self.text_ln = nn.LayerNorm(d_model)

        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def encode_visual(self, frames: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = frames.shape
        x = frames.view(b * t, c, h, w)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.visual_pos[:, : x.size(1)]
        x = self.visual_encoder(x)
        x = self.visual_ln(x)
        return x.view(b, t, x.size(1), x.size(2))

    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.text_embed(token_ids)
        x = x + self.text_pos[:, : x.size(1)]
        key_padding_mask = token_ids.eq(0)
        x = self.text_encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.text_ln(x)
        return x


class CaTeRMini(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        vocab_size: int = 8192,
        image_size: int = 96,
        patch_size: int = 16,
        max_text_len: int = 77,
        max_frames: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = FrozenDualEncoder(
            d_model=d_model,
            vocab_size=vocab_size,
            image_size=image_size,
            patch_size=patch_size,
            max_text_len=max_text_len,
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
        return logits, h_video
