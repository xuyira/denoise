import torch
from torch import nn


class EEGANetResidualBlock(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels, momentum=0.5),
            nn.PReLU(num_parameters=channels),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels, momentum=0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class GanGenerator(nn.Module):
    """EEGANet-style residual generator adapted from 2D Conv to 1D EEG epochs."""

    def __init__(self, hidden: int = 64, num_blocks: int = 16, use_tanh: bool = False):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=9, padding=4),
            nn.PReLU(num_parameters=hidden),
        )
        self.residual = nn.Sequential(*[EEGANetResidualBlock(hidden) for _ in range(num_blocks)])
        self.trunk = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden, momentum=0.5),
        )
        self.exit = nn.Conv1d(hidden, 1, kernel_size=9, padding=4)
        self.output_activation = nn.Tanh() if use_tanh else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.entry(x)
        x = features + self.trunk(self.residual(features))
        x = self.exit(x)
        return self.output_activation(x)


class GanDiscriminator(nn.Module):
    """EEGANet-style discriminator adapted from 2D Conv to 1D EEG epochs."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden, momentum=0.5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden, hidden * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden * 2, momentum=0.5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden * 2, hidden * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden * 2, momentum=0.5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden * 2, hidden * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden * 4, momentum=0.5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden * 4, hidden * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden * 4, momentum=0.5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden * 4, hidden * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden * 8, momentum=0.5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden * 8, hidden * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden * 8, momentum=0.5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(2),
            nn.Flatten(),
            nn.Linear(hidden * 16, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def gan_losses(real_logits: torch.Tensor, fake_logits: torch.Tensor):
    bce = nn.BCEWithLogitsLoss()
    real_targets = torch.ones_like(real_logits)
    fake_targets = torch.zeros_like(fake_logits)
    d_loss = bce(real_logits, real_targets) + bce(fake_logits, fake_targets)
    g_adv_loss = bce(fake_logits, real_targets)
    return g_adv_loss, d_loss


def build_gan_models(hidden: int = 64, num_blocks: int = 16, use_tanh: bool = False):
    return GanGenerator(hidden=hidden, num_blocks=num_blocks, use_tanh=use_tanh), GanDiscriminator(hidden=hidden)
