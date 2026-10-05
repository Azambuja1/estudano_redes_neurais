import torch
import numpy as np
from torch import nn

class LineNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1,1)
                                   )
    def forward(self, x):
        return self.layers(x)