import torch
from models.utils import FeedFoward

class Block(torch.nn.Module):
    def __init__(self, emb_dim, num_heads, head_size, dropout = 0.1):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(emb_dim)
        self.sa_head = MultiHeadAttention(emb_dim=emb_dim, 
                                          head_size=head_size, 
                                          num_heads=num_heads, 
                                          block_size=None, 
                                          dropout=dropout)
        self.mlp = FeedFoward(emb_dim, 'gelu', dropout)
        self.ln1 = torch.nn.LayerNorm(emb_dim)
        self.ln2 = torch.nn.LayerNorm(emb_dim)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

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

    def forward(self, q, k, v):

        k = self.key(k)
        q = self.query(q)
        v = self.value(v)

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

    def forward(self, q, k, v):
        out = torch.cat([h.forward(q, k, v) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out