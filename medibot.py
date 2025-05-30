from utils import TextProcessor, FaissIndex, SentenceEmbedder
from groq import Groq

class MediBot:
    """A virtual assistant for medical queries"""

    def __init__(self):
        """
        Initializes the MediBot with necessary components.
        """
        # Initialize the TextProcessor and read the corpus
        self.processor = TextProcessor()
        self.corpus = self.processor.create_corpus('faq.txt')

        # Initialize the SentenceEmbedder
        self.embedder = SentenceEmbedder()
        embeddings = self.embedder.generate_embeddings(self.corpus)

        # Create a FAISS index
        self.index = FaissIndex(dimension=len(embeddings[1]))
        self.index.add_embeddings(embeddings)

        # Initialize the Groq
        self.groq = Groq()
        self.groq.initialize()

    def generate_response(self, query):
        """
        Generates a response for the given query using the Groq model.

        :param query: The query string to generate a response for.
        :return: The generated response from the Groq model.
        """
        query_embedding = self.embedder.generate_embeddings([query])[0]
        indices, _ = self.index.search(query_embedding, k=5)

        # Add nearest sentences to the Groq model
        self.groq.add_nearest_sentences(self.corpus, indices)

        # Add user message to the Groq model
        self.groq.user_message(query)
        
        # Generate a response from the Groq model
        response = self.groq.generate_response()
        
        return response.get("content", "No response generated.")
    
    def reset_conversation(self):
        """
        Resets the conversation state in the Groq model.
        """
        self.groq.reset()