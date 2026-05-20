import torch
from torch.nn import functional as F
from transformers.attention import *
from transformers.simple_model import FeedFoward

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
        x = x + self.sa_head(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class VisionTransformer(torch.nn.Module):
    def __init__(self, emb_dim, patch_size, n_patches, num_heads, head_size, n_layer, dropout):
        super().__init__()

        self.patch_projection = torch.nn.Linear(patch_size**2, emb_dim)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.position_embedding_table = torch.nn.Embedding(n_patches + 1, emb_dim)
        self.blocks = torch.nn.Sequential(
            *[Block(emb_dim, num_heads, head_size, dropout) for _ in range(n_layer)]
        )

        self.ln = torch.nn.LayerNorm(emb_dim)
        self.classifier = torch.nn.Linear(emb_dim, 1)

    def forward(self, patches, targets):
        B, N, P_2 = patches.shape

        patch_embeddings = self.patch_projection(patches)  # (B, N, emb_dim)

        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, emb_dim)

        x = torch.cat([cls_token, patch_embeddings], dim=1)  # (B, N + 1, emb_dim)

        position_indices = torch.arange(N + 1, device=patches.device)
        position_embeddings = self.position_embedding_table(position_indices)  # (N + 1, emb_dim)

        x = x + position_embeddings.unsqueeze(0)  # (B, N + 1, emb_dim)

        x = self.blocks(x)
        x = self.ln(x)

        cls_output = x[:, 0, :]  # (B, emb_dim)

        logits = self.classifier(cls_output)  # (B, 1)
        logits = logits.squeeze(-1)  # (B,)
        loss = F.binary_cross_entropy_with_logits(logits, targets)

        return logits, loss
    
    def predict(self, patches, threshold=0.5):
        logits, _ = self(patches)
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()
        return preds