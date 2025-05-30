import re
import faiss
import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

class FaissIndex:
    """
    A class to handle FAISS index operations for sentence embeddings.
    """
    
    def __init__(self, dimension=384):
        """
        Initializes the FAISS index with the specified embedding dimension.
        
        :param dimension: Dimension of the embeddings.
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
    
    def add_embeddings(self, embeddings):
        """
        Adds embeddings to the FAISS index.
        
        :param embeddings: List of embeddings to be added.
        """
        embedding_matrix = np.array(embeddings).astype('float32')
        self.index.add(embedding_matrix)
    
    def search(self, query_embedding, k=5):
        """
        Searches the FAISS index for the k nearest neighbors of the query embedding.
        
        :param query_embedding: Embedding of the query sentence.
        :param k: Number of nearest neighbors to return.
        :return: Indices and distances of the k nearest neighbors.
        """
        query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)
        return indices[0].tolist(), distances[0].tolist()
    

class SentenceEmbedder:
    """
    A class to handle sentence embedding operations using a pre-trained model.
    """
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initializes the SentenceEmbedder with the specified model.
        
        :param model_name: Name of the pre-trained SentenceTransformer model.
        """
        self.model = SentenceTransformer(model_name)
    
    def generate_embeddings(self, sentences):
        """
        Generates embeddings for a list of sentences.
        
        :param sentences: List of sentences to be embedded.
        :return: List of embeddings corresponding to the input sentences.
        """
        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        return embeddings.tolist()  # Convert to list for easier handling
    

class TextProcessor:
    """
    A class to handle text processing operations such as reading files and cleaning text.
    """
    
    @staticmethod
    def clean_text(text):
        """
        Cleans the input text by removing extra spaces and newlines.
        
        :param text: Input text to be cleaned.
        :return: Cleaned text.
        """
        # Remove extra spaces and newlines
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        return cleaned_text
    
    def split_text_into_sentences(self, text):
        """
        Splits the input text into sentences using NLTK's sentence tokenizer.
        
        :param text: Input text to be split into sentences.
        :return: List of sentences.
        """
        return sent_tokenize(text)
    
    def create_corpus(self, file_path):
        """
        Reads a file and returns its content as a cleaned string.
        
        :param file_path: Path to the file to be read.
        :return: List of sentences from the cleaned text.
        """
        with open(file_path, 'r') as file:
            text = file.read()
        return self.split_text_into_sentences(self.clean_text(text))
    