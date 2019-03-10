import torch
import torch.nn as nn
import torch.nn.functional as F

class Baseline(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        pass