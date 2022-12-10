import torch
import torch.nn as nn

from .attn import Attention
from .conv import ConformerConvModule
from .feed_forward import FeedForward
from .wrappers import PreNorm, Scale

from .params import (
    hist_frame as default_hist_frame,
    look_ahead as default_look_ahead
)



class ConformerBlock(nn.Module):
    def __init__(self,
                 *,
                 dim,
                 dim_head=64,
                 heads=8,
                 ff_mult=4,
                 conv_expansion_factor=2,
                 conv_kernel_size=31,
                 attn_dropout=0.,
                 ff_dropout=0.,
                 conv_dropout=0,
                 hist_frame=default_hist_frame,
                 look_ahead=default_look_ahead
                 ):
        super(ConformerBlock, self).__init__()
        self.head = heads
        self.hist_frame = hist_frame
        self.look_ahead = look_ahead
        self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.attn = Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)
        self.conv = ConformerConvModule(dim=dim, casual=False, expansion_factor=conv_expansion_factor,
                                        kernel_size=conv_kernel_size, dropout=conv_dropout)
        self.ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)

        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        x = self.ff1(x) + x
        x = self.attn(x, mask=mask) + x
        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x


