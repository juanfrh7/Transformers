import nltk
import tiktoken
import torch
import torch.nn as nn
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
    def __init__(self, embedding_type="gpt2", embedding_dim=50, seed=42):
        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim
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
            self.embedding_layer = nn.Embedding(vocab_size, self.embedding_dim)
            # X = self.embedding_layer(ids)  # (L, D)
            return self.embedding_layer

    def decode_ids(self, encoding):
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