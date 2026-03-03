import nltk
import tiktoken
import torch.nn as nn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec


nltk.download('punkt_tab')
nltk.download('stopwords') 

class word_embedding():
    def __init__(self, embedding_type = "gpt2", embedding_dim = 50):

        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim

    def tokenize(self, text):
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        return tokens

    def create_embedding(self, text):

        if self.embedding_type == "word2vec":
            tokens = self.tokenize(text)
            self.embedding_model = Word2Vec(sentences=tokens, 
                            vector_size=self.embedding_dim, 
                            min_count=1, seed=42)
            embedding = self.embedding_model.wv

            return embedding
            
        elif self.embedding_type == "gpt2":
            encoding_model = tiktoken.get_encoding("gpt2")
            encoding = encoding_model.encode(text)
            embedding = nn.Embedding(len(encoding), self.embedding_dim)
            
            return encoding, embedding
    
    def decode(self, embedding):

        if self.embedding_type == "word2vec":
            # vector can be a numpy array or a torch tensor
            v = embedding
            if hasattr(v, "detach"):  # torch tensor
                v = v.detach().cpu().numpy()

            word, score = self.embedding_model.most_similar(positive=[v], topn=1)[0]
            return word, score
        
        elif self.embedding_type == "gpt2":
            return self.embedding_model.decode(embedding)