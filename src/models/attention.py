import torch
from models.utils import FeedFoward

class GWTBlock(torch.nn.Module):
    def __init__(self, emb_dim, ffn_dim, num_heads, head_size_k, head_size_v, dropout = 0.1):
        super().__init__()

        self.num_heads = num_heads
        self.head_size_k = head_size_k
        self.head_size_v = head_size_v

        self.ln_x_write = torch.nn.LayerNorm(emb_dim)
        self.ln_workspace_write = torch.nn.LayerNorm(emb_dim)

        self.ln_x_broadcast = torch.nn.LayerNorm(emb_dim)
        self.ln_workspace_broadcast = torch.nn.LayerNorm(emb_dim)

        self.ln_x_mlp = torch.nn.LayerNorm(emb_dim)
        self.mlp = FeedFoward(emb_dim, ffn_dim, 'gelu', dropout)

        self.write_attention = MultiHeadAttention(
            emb_dim=emb_dim,
            head_size_k=head_size_k,
            head_size_v=head_size_v,
            num_heads=num_heads,
            dropout=dropout
        )

        self.broadcast_attention = MultiHeadAttention(
            emb_dim=emb_dim,
            head_size_k=head_size_k,
            head_size_v=head_size_v,
            num_heads=num_heads,
            dropout=dropout
        )

        self.input_proj = torch.nn.Linear(emb_dim, emb_dim, bias=False)
        self.input_gate = torch.nn.Linear(emb_dim, emb_dim, bias=True)
        self.forget_gate = torch.nn.Linear(emb_dim, emb_dim, bias=True)

    def forward(self, x, workspace):

        # Write step: patch agents write into workspace
        patch_agents = x[:, 1:, :]   # only patches can write, cls token does not write

        workspace_update = self.write_attention(
            q=self.ln_workspace_write(workspace),
            k=self.ln_x_write(patch_agents),
            v=self.ln_x_write(patch_agents)
        )

        input_gate, forget_gate = self.compute_gwt_gates(patch_agents, workspace)

        workspace = input_gate * torch.tanh(workspace_update) + forget_gate * workspace

        # Broadcast step: workspace broadcasts back to all patch agents
        x_update = self.broadcast_attention(
            q=self.ln_x_broadcast(x),
            k=self.ln_workspace_broadcast(workspace),
            v=self.ln_workspace_broadcast(workspace)
        )

        x = x + x_update  # (B, N + 1, emb_dim)
        x = x + self.mlp(self.ln_x_mlp(x))
        return x, workspace
    
    def compute_gwt_gates(self, patch_agents, workspace):

        X = torch.relu(self.input_proj(patch_agents))
        X = X.mean(dim=1)
        X = X.unsqueeze(1)   # [B, 1, emb_dim]

        K = X + torch.tanh(workspace)   # [B, M, emb_dim]

        I = torch.sigmoid(self.input_gate(K))
        F = torch.sigmoid(self.forget_gate(K)) # [B, M, emb_dim]

        return I, F
    

class Block(torch.nn.Module):
    def __init__(self, emb_dim, ffn_dim, num_heads, head_size_k, head_size_v, dropout = 0.1):
        super().__init__()

        self.sa_head = MultiHeadAttention(emb_dim=emb_dim, 
                                          head_size_k=head_size_k,
                                          head_size_v=head_size_v,
                                          num_heads=num_heads, 
                                          block_size=None, 
                                          dropout=dropout)
        self.mlp = FeedFoward(emb_dim, ffn_dim, 'gelu', dropout)
        self.ln1 = torch.nn.LayerNorm(emb_dim)
        self.ln2 = torch.nn.LayerNorm(emb_dim)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class Head(torch.nn.Module):
    """ one head of self attention """

    def __init__(self, emb_dim, head_size_k, head_size_v, block_size = None, dropout=0.1):
        super().__init__()
        self.key = torch.nn.Linear(emb_dim, head_size_k, bias=False)
        self.query = torch.nn.Linear(emb_dim, head_size_k, bias=False)
        self.value = torch.nn.Linear(emb_dim, head_size_v, bias=False)
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

    def __init__(self, emb_dim, head_size_k, head_size_v, num_heads, block_size = None, dropout=0.1):
        super().__init__()
        self.heads = torch.nn.ModuleList([Head(emb_dim, head_size_k, head_size_v, block_size, dropout) for _ in range(num_heads)])
        self.proj = torch.nn.Linear(head_size_v * num_heads, emb_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, q, k, v):
        out = torch.cat([h(q, k, v) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
