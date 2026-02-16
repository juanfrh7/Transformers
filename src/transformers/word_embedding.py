import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

nltk.download('punkt_tab')
nltk.download('stopwords') 

class word_embedding():
    def __init__(self, text, embedding_dim = 50):
        self.original_text = text
        self.embedding_dim = embedding_dim
        self.tokens = self.tokenize(text)
        self.embedding_model = self.create_embedding(self.tokens)

    def tokenize(self, text):
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        return tokens

    def create_embedding(self, tokens):
        model = Word2Vec(sentences=tokens, 
                         vector_size=self.embedding_dim, 
                         min_count=1, seed=42)
        return model
    
    def most_similar_word(self, vector):
        # vector can be a numpy array or a torch tensor
        v = vector
        if hasattr(v, "detach"):  # torch tensor
            v = v.detach().cpu().numpy()

        word, score = self.embedding_model.most_similar(positive=[v], topn=1)[0]
        return word, score