import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(42)

class FeedFoward(torch.nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, layer = 'relu', dropout = 0.1):
        super().__init__()

        if layer == 'relu':
            l = torch.nn.ReLU()

        elif layer == 'gelu':
            l = torch.nn.GELU()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_embd, 2 * n_embd),
            l,
            torch.nn.Linear(2 * n_embd, n_embd),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class BigramLanguageModel(nn.Module):

    def __init__(self, embedding):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = embedding

    def forward(self, idx, targets=None):

        logits = self.token_embedding_table(idx) # (B,T,C) - batch, sequence length, vocab size/emd_dim

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx