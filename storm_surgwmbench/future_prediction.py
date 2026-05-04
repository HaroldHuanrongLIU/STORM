from __future__ import annotations

import sys
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if (parent / "surgwmbench_benchmark").is_dir():
        sys.path.insert(0, str(parent))
        break

import torch
from torch import nn

from storm_surgwmbench.adapter import SurgWMBenchStormTransformer
from surgwmbench_benchmark.future_model_helpers import normalized_context_time, normalized_future_time
from surgwmbench_benchmark.future_prediction import FutureProtocolConfig, main


class STORMFuturePredictionModel(nn.Module):
    """Future-prediction wrapper around the STORM stochastic Transformer core."""

    def __init__(self, config: FutureProtocolConfig) -> None:
        super().__init__()
        self.core = SurgWMBenchStormTransformer(
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            num_layers=2,
            num_heads=4,
        )
        self.hidden_to_latent = nn.Linear(config.hidden_dim, config.latent_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        frames = batch["context_frames"]
        bsz, context, channels, height, width = frames.shape
        z, _, _ = self.core.encoder(frames.reshape(bsz * context, channels, height, width))
        context_z = z.view(bsz, context, -1)
        future_z = context_z[:, -1:].expand(-1, batch["future_frame_indices"].shape[1], -1)
        model_in = torch.cat(
            [
                self.core.input_proj(torch.cat([context_z, normalized_context_time(batch)], dim=-1)),
                self.core.input_proj(torch.cat([future_z, normalized_future_time(batch)], dim=-1)),
            ],
            dim=1,
        )
        hidden = self.core.transformer(model_in)[:, context:]
        pred_coords = torch.sigmoid(self.core.coord_head(hidden))
        z_future = self.hidden_to_latent(hidden)
        pred_frames = self.core.decoder(z_future.reshape(-1, z_future.shape[-1]), (height, width))
        pred_frames = pred_frames.view(bsz, z_future.shape[1], channels, height, width)
        return {"pred_frames": pred_frames, "pred_coords_norm": pred_coords}


def make_model(config: FutureProtocolConfig) -> nn.Module:
    return STORMFuturePredictionModel(config)


if __name__ == "__main__":
    raise SystemExit(main("storm", "STORMFuturePredictionCore", "storm_surgwmbench.data.surgwmbench", make_model))
