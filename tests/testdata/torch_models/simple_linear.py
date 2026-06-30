import torch
import torch.nn as nn

class SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 4, bias=True)
    def forward(self, x):
        return self.fc(x)
