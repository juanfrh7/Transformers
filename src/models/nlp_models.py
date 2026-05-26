import torch
from torch.nn import functional as F
from .attention import MultiHeadAttention
from .utils import FeedFoward

import nltk
import tiktoken
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')
nltk.download('stopwords') 

def tokenize_word(text):
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text.lower())
    tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
    return tokens

class word_embedding:
    def __init__(self, embedding_type="gpt2", seed=42):
        self.embedding_type = embedding_type
        self.seed = seed

        self.w2v_model = None
        self.tokenizer = None
        self.embedding_layer = None

    def encode(self, text):
        """
        Returns discrete tokens/ids (encoding).
        """
        if self.embedding_type == "word2vec":
            return self.tokenize_word(text)              # list[str]
        elif self.embedding_type == "gpt2":
            self.tokenizer = tiktoken.get_encoding("gpt2")
            return self.tokenizer.encode(text)           # list[int]

    def embed(self, encoding):
        """
        Returns continuous vectors (embeddings) for the given encoding.
        """
        if self.embedding_type == "word2vec":
            tokens = encoding
            self.w2v_model = Word2Vec(
                sentences=[tokens],
                vector_size=self.embedding_dim,
                min_count=1,
                seed=self.seed
            )
            wv = self.w2v_model.wv
            X = torch.stack([torch.tensor(wv[t], dtype=torch.float32) for t in tokens], dim=0)  # (L, D)
            return X

        elif self.embedding_type == "gpt2":
            # ids = torch.tensor(encoding, dtype=torch.long)
            if self.tokenizer is None:
                self.tokenizer = tiktoken.get_encoding("gpt2")
            vocab_size = self.tokenizer.n_vocab

            # Random embedding layer (NOT pretrained GPT-2)
            self.embedding_layer = torch.nn.Embedding(vocab_size, vocab_size)
            # X = self.embedding_layer(ids)  # (L, D)
            return self.embedding_layer

    def decode(self, encoding):
        """
        Decodes ONLY discrete token ids back to text (GPT-2 case).
        """
        if self.embedding_type != "gpt2":
            raise ValueError("decode_ids is only for gpt2 token ids.")
        return self.tokenizer.decode(encoding)

    def nearest_word_word2vec(self, vector, topn=1):
        """
        Approx 'decode' a vector -> nearest word (Word2Vec case).
        """
        if self.embedding_type != "word2vec" or self.w2v_model is None:
            raise ValueError("Word2Vec model not trained yet.")
        v = vector.detach().cpu().numpy() if hasattr(vector, "detach") else vector
        return self.w2v_model.wv.most_similar(positive=[v], topn=topn)
    

class Block(torch.nn.Module):
    def __init__(self, vocab_dim, num_heads, block_size, dropout):
        super().__init__()
        
        self.head_dim = vocab_dim // num_heads
        self.sa_head = MultiHeadAttention(vocab_dim, self.head_dim, num_heads, block_size, dropout) # self-attention head
        self.ffwd = FeedFoward(vocab_dim, 'relu', dropout) # feed-forward network
        self.ln1 = torch.nn.LayerNorm(vocab_dim)
        self.ln2 = torch.nn.LayerNorm(vocab_dim)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x), self.ln1(x), self.ln1(x))
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
    

class BigramLanguageModel(torch.nn.Module):

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