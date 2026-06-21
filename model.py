import torch.nn as nn

class FraudNet(nn.Module):
    def __init__(self, input_dim=29):
        super(FraudNet, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),   # 🔥 important
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),   # 🔥 important
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.fc(x)