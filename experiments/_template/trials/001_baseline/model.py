"""Trial 001 model: a tiny CNN policy stub. Replace the TODOs with real logic.

Crafter observations are uint8 images of shape (64, 64, 3); the action space is
Discrete. This is a minimal Nature-CNN-style encoder + linear policy head to
copy and grow -- it is NOT trained or tuned.
"""
import torch
import torch.nn as nn


class Policy(nn.Module):
    def __init__(self, n_actions: int, in_channels: int = 3):
        super().__init__()
        # TODO: tune the architecture. This is a placeholder Nature-CNN encoder.
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        # 64x64 input -> encoder output dim (compute once, lazily).
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 64, 64)
            feat_dim = self.encoder(dummy).shape[1]
        self.policy_head = nn.Linear(feat_dim, n_actions)
        # TODO: add a value head for actor-critic methods.

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, H, W, C) uint8 -> action logits (B, n_actions)."""
        # TODO: handle batching/normalisation properly in the training loop.
        x = obs.float() / 255.0
        if x.dim() == 4 and x.shape[-1] in (1, 3):  # NHWC -> NCHW
            x = x.permute(0, 3, 1, 2)
        return self.policy_head(self.encoder(x))
