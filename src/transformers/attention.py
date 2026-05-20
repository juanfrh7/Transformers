import torch
from torch.nn import functional as F

class Head(torch.nn.Module):
    """ one head of self attention """

    def __init__(self, emb_dim, head_size, block_size = None, dropout=0.1):
        super().__init__()
        self.key = torch.nn.Linear(emb_dim, head_size, bias=False)
        self.query = torch.nn.Linear(emb_dim, head_size, bias=False)
        self.value = torch.nn.Linear(emb_dim, head_size, bias=False)
        self.block_size = block_size
        if self.block_size is not None:
            self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        weight = torch.matmul(q, k.transpose(-2, -1)) / k.shape[-1]**0.5
        
        if self.block_size is not None:
            T = weight.shape[-1]
            weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weight = torch.softmax(weight, dim=-1)
        weight = self.dropout(weight)

        out = weight @ v

        return out

class MultiHeadAttention(torch.nn.Module):
    """ multiple heads of self attention in parallel """

    def __init__(self, emb_dim, head_size, num_heads, block_size = None, dropout=0.1):
        super().__init__()
        self.heads = torch.nn.ModuleList([Head(emb_dim, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = torch.nn.Linear(head_size * num_heads, emb_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out