from attention import *


class GPTLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size, vocab_dim, block_size, num_heads):
        super().__init__()
        self.token_embedding_table = torch.nn.Embedding(vocab_size, vocab_dim) # holds semantic value
        self.position_embedding_table = torch.nn.Embedding(1000, vocab_dim) 
        self.lm_head = torch.nn.Linear(vocab_dim, vocab_size)
        self.sa_head = MultiHeadAttention(vocab_dim, vocab_dim // num_heads, num_heads)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embeddings = self.token_embedding_table(idx) # (B,T,C)
        position_embeddings = self.position_embedding_table(torch.arange(T)) # (T,C)
        x = token_embeddings + position_embeddings # (B,T,C)
        x = self.sa_head(x)
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
            idx_cond = idx[:, -block_size:]
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

