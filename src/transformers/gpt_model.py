import torch
from torch.nn import functional as F
from .attention import *
from .simple_model import FeedFoward

class Block(torch.nn.Module):
    def __init__(self, vocab_dim, num_heads, block_size, dropout):
        super().__init__()
        
        self.head_dim = vocab_dim // num_heads
        self.sa_head = MultiHeadAttention(vocab_dim, self.head_dim, num_heads, block_size, dropout) # self-attention head
        self.ffwd = FeedFoward(vocab_dim, 'relu', dropout) # feed-forward network
        self.ln1 = torch.nn.LayerNorm(vocab_dim)
        self.ln2 = torch.nn.LayerNorm(vocab_dim)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size, vocab_dim, block_size, num_heads, n_layer, dropout):
        super().__init__()
        self.block_size = block_size
        self.num_heads = num_heads
        self.n_layer = n_layer

        self.token_embedding_table = torch.nn.Embedding(vocab_size, vocab_dim) # holds semantic value
        self.position_embedding_table = torch.nn.Embedding(block_size, vocab_dim) # holds positional value (up to 1000 tokens)
        self.blocks = torch.nn.Sequential(*[Block(vocab_dim, num_heads, block_size, dropout) for _ in range(n_layer)])
        self.fln = torch.nn.LayerNorm(vocab_dim)
        self.lm_head = torch.nn.Linear(vocab_dim, vocab_size) # language modeling head to produce logits for each token in the vocab

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embeddings = self.token_embedding_table(idx) # (B,T,C)
        position_embeddings = self.position_embedding_table(torch.arange(T)) # (T,C)
        x = token_embeddings + position_embeddings # (B,T,C)
        x = self.blocks(x)
        x = self.fln(x)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = torch.nn.functional.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx