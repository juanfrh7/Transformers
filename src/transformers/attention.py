import torch
from torch.nn import functional as F

class Head(torch.nn.Module):
    """ one head of self attention """

    def __init__(self, vocab_dim, head_size):
        super().__init__()
        self.key = torch.nn.Linear(vocab_dim, head_size, bias=False)
        self.query = torch.nn.Linear(vocab_dim, head_size, bias=False)
        self.value = torch.nn.Linear(vocab_dim, head_size, bias=False)

    def forward(self, x):

        B, T, C = x.shape
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        v = self.value(x) # (B,T,head_size)

        head_size = k.shape[-1]

        weight = torch.matmul(q, k.transpose(-2, -1)) / k.shape[-1]**0.5
        mask = torch.triu(torch.ones(head_size, head_size, device=weight.device, dtype=torch.bool), diagonal=1)
        weight = weight.masked_fill(mask, float("-inf"))
        weight = torch.softmax(weight, dim=-1)

        out = weight @ v

        return out

class MultiHeadAttention(torch.nn.Module):
    """ multiple heads of self attention in parallel """

    def __init__(self, vocab_dim, head_size, num_heads):
        super().__init__()
        self.heads = torch.nn.ModuleList([Head(vocab_dim, head_size) for _ in range(num_heads)])
        self.proj = torch.nn.Linear(head_size * num_heads, vocab_dim)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out