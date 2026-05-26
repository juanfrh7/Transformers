import torch
from torch.nn import functional as F
from models.attention import Block, MultiHeadAttention

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


class VisionTransformerGWT(torch.nn.Module):
    def __init__(
        self,
        emb_dim,
        patch_size,
        n_patches,
        num_heads,
        head_size,
        num_workspace_slots,
        dropout=0.1
    ):
        super().__init__()

        self.emb_dim = emb_dim
        self.n_patches = n_patches
        self.num_workspace_slots = num_workspace_slots

        self.patch_projection = torch.nn.Linear(patch_size ** 2, emb_dim)
        self.position_embedding_table = torch.nn.Embedding(n_patches + 1, emb_dim)

        self.global_workspace = torch.nn.Parameter(
            torch.randn(1, num_workspace_slots, emb_dim)
        )

        self.cls_token = torch.nn.Parameter(
            torch.randn(1, 1, emb_dim)
        )

        self.ln_x_write = torch.nn.LayerNorm(emb_dim)
        self.ln_workspace_write = torch.nn.LayerNorm(emb_dim)

        self.ln_x_broadcast = torch.nn.LayerNorm(emb_dim)
        self.ln_workspace_broadcast = torch.nn.LayerNorm(emb_dim)

        self.write_attention = MultiHeadAttention(
            emb_dim=emb_dim,
            head_size=head_size,
            num_heads=num_heads,
            dropout=dropout
        )

        self.broadcast_attention = MultiHeadAttention(
            emb_dim=emb_dim,
            head_size=head_size,
            num_heads=num_heads,
            dropout=dropout
        )

        self.ln_final = torch.nn.LayerNorm(emb_dim)
        self.classifier = torch.nn.Linear(emb_dim, 1)

    def forward(self, patches, targets=None):
        B, N, P_2 = patches.shape

        x = self.patch_projection(patches)  # (B, N, emb_dim)

        position_indices = torch.arange(N, device=patches.device)
        position_embeddings = self.position_embedding_table(position_indices)

        x = x + position_embeddings.unsqueeze(0)  # (B, N + 1, emb_dim)

        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        workspace = self.global_workspace.expand(B, -1, -1)  # (B, M, emb_dim)

        # Write step: patch agents write into workspace
        workspace_update = self.write_attention.forward(
            q=self.ln_workspace_write(workspace),
            k=self.ln_x_write(x),
            v=self.ln_x_write(x)
        )

        workspace = workspace + workspace_update  # (B, M, emb_dim)

        # Broadcast step: workspace broadcasts back to all patch agents
        x_update = self.broadcast_attention.forward(
            q=self.ln_x_broadcast(x),
            k=self.ln_workspace_broadcast(workspace),
            v=self.ln_workspace_broadcast(workspace)
        )

        x = x + x_update  # (B, N + 1, emb_dim)

        x = self.ln_final(x)

        # one image-level representation
        cls_output = x[:, 0, :]  # (B, emb_dim)

        logits = self.classifier(cls_output)  # (B, 1)
        logits = logits.squeeze(-1)  # (B,)

        if targets is None:
            loss = None
        else:
            targets = torch.as_tensor(
                targets,
                device=logits.device,
                dtype=torch.float32
            ).reshape(-1)

            loss = F.binary_cross_entropy_with_logits(logits, targets)

        return logits, loss

    @torch.no_grad()
    def predict(self, patches, threshold=0.5):
        self.eval()

        logits, _ = self(patches)
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        return probs, preds