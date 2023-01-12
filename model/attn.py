from einops import rearrange

import numpy as np
import torch
import torch.nn as nn
from torch import einsum


from .utils import default, exists

def gen_mask(look_ahead, hist_frame, T, device='cude'):

    mask = np.zeros((T,T), dtype=np.float32)

    n_chunk = T // look_ahead

    for i in range(n_chunk + 1):
        start = i * look_ahead
        lt_x = start
        lt_y = np.maximum(0, start - hist_frame)
        rt_y = start + look_ahead
        lb_x = np.minimum(T, lt_x + look_ahead)
        mask[lt_x:lb_x, lt_y:rt_y] = 1

    return torch.from_numpy(mask).to(device)

NINF = float('inf')

class AttentionStream(nn.Module):
    def __init__(self,
                 dim,
                 heads=8,
                 dim_head=64,
                 dropout=0.,
                 max_pos_emb=512):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)

        self.droupout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None, context_mask=None):
        n, device, h, max_pos_emb, has_content = x.shape[-2], x.device, self.heads, self.max_pos_emb, exists(context)
        context = default(context, x)

        q, (k, v) = self.to_q(x), self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        dots = einsum('b h n d, b h r d -> b h n r', q, k) * self.scale

        seq = torch.arange(n, device=device)
        dist = rearrange(seq, 'i -> i ()') - rearrange(seq, 'j -> () j')
        dist = torch.clamp(dist, -max_pos_emb, max_pos_emb) + max_pos_emb
        rel_pos_emd = self.rel_pos_emb(dist).to(q)
        pos_attn = einsum('b h n d, n r d -> b h n r', q, rel_pos_emd) * self.scale
        dots = dots + pos_attn
        if mask is not None:
            mask_value = -torch.finfo(dots.dtype).max
            dots.masked_fill_(~mask, mask_value)
        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.droupout(out)


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 heads=8,
                 dim_head=64,
                 dropout=0.,
                 max_pos_emb=512):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)

        self.droupout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None, context_mask=None):
        n, device, h, max_pos_emb, has_content = x.shape[-2], x.device, self.heads, self.max_pos_emb, exists(context)
        context = default(context, x)

        q, (k, v) = self.to_q(x), self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        dots = einsum('b h n d, b h r d -> b h n r', q, k) * self.scale

        seq = torch.arange(n, device=device)
        dist = rearrange(seq, 'i -> i ()') - rearrange(seq, 'j -> () j')
        dist = torch.clamp(dist, -max_pos_emb, max_pos_emb) + max_pos_emb
        rel_pos_emd = self.rel_pos_emb(dist).to(q)
        pos_attn = einsum('b h n d, n r d -> b h n r', q, rel_pos_emd) * self.scale
        dots = dots + pos_attn

        if exists(mask) or exists(context_mask):
            mask = default(mask, lambda: torch.ones(*x.shape[:2], device=device))
            context_mask = default(context_mask, mask) if not has_content else default(context_mask, lambda: torch.ones(*context.shape[:2], device=device))
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(context_mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.droupout(out)






