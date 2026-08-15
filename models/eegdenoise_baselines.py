import torch
from torch import nn


class FcNN(nn.Module):
    def __init__(self, datanum: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(datanum, datanum),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(datanum, datanum),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(datanum, datanum),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(datanum, datanum),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(1)
        x = self.net(x)
        return x.unsqueeze(1)


class RNNLSTM(nn.Module):
    def __init__(self, datanum: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=1, batch_first=True)
        self.net = nn.Sequential(
            nn.Linear(datanum, datanum),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(datanum, datanum),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(datanum, datanum),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x.flatten(1)
        x = self.net(x)
        return x.unsqueeze(1)


class SimpleCNN(nn.Module):
    def __init__(self, datanum: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.head = nn.Linear(64 * datanum, datanum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        x = self.head(x)
        return x.unsqueeze(1)


class ResBasicBlock(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + x


class BasicBlockAll(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch3 = nn.Sequential(ResBasicBlock(3), ResBasicBlock(3))
        self.branch5 = nn.Sequential(ResBasicBlock(5), ResBasicBlock(5))
        self.branch7 = nn.Sequential(ResBasicBlock(7), ResBasicBlock(7))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.branch3(x), self.branch5(x), self.branch7(x)], dim=1)


class ComplexCNN(nn.Module):
    def __init__(self, datanum: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            BasicBlockAll(),
            nn.Conv1d(96, 32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.head = nn.Linear(32 * datanum, datanum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = x.flatten(1)
        x = self.head(x)
        return x.unsqueeze(1)


def build_baseline_model(name: str, datanum: int) -> nn.Module:
    key = name.lower()
    if key == "fcnn":
        return FcNN(datanum)
    if key == "simple_cnn":
        return SimpleCNN(datanum)
    if key == "complex_cnn":
        return ComplexCNN(datanum)
    if key == "rnn_lstm":
        return RNNLSTM(datanum)
    raise ValueError(f"Unsupported baseline model: {name}")
