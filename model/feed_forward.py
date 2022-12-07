import torch.nn as nn
from .act import Swish

class FeedForward(nn.Module):
    def __init__(self,
                 dim,
                 mult=4,
                 dropout=0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)