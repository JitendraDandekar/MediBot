import os
import requests

class Groq:
    """
    A class to handle Groq model operations.
    """
    url = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, model_name="llama-3.3-70b-versatile"):
        """
        Initializes the Groq model with the specified model name.

        :param model_name: Name of the Groq model.
        """
        self.model_name = model_name
        
        self.api_key = os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        self.messages = []

    def initialize(self):
        """
        Initializes the Groq model with system messages.
        This method sets up the initial system messages that guide the model's behavior.
        """
        self.system_message("You are a helpful assistant.")
        self.system_message("Please provide a detailed answer based on the nearest sentences.")
        self.system_message("Use the following sentences to answer the user's query:")
    
    def add_nearest_sentences(self, corpus, indices):
        """
        Adds the nearest sentences to the Groq model's messages.

        :param corpus: A list of sentences that are relevant to the user's query.
        :param indices: A list of indices corresponding to the nearest sentences in the corpus.
        """
        if not isinstance(corpus, list) or not isinstance(indices, list):
            raise ValueError("Both corpus and indices must be lists.")
        
        if not all(isinstance(idx, int) for idx in indices):
            raise ValueError("All indices must be integers.")
        
        if not all(0 <= idx < len(corpus) for idx in indices):
            raise ValueError("Indices must be within the range of the corpus length.")
        
        for idx in indices:
            self.system_message(corpus[idx])

    def message(self, content, role="user"):
        """
        Creates a message dictionary for the Groq model.

        :param content: The content of the message.
        :param role: The role of the message sender (default is "user").
        :return: A dictionary representing the message.
        """
        return self.__add_message({
            "role": role,
            "content": content
        })
    
    def system_message(self, content):
        """
        Creates a system message dictionary for the Groq model.

        :param content: The content of the system message.
        :return: A dictionary representing the system message.
        """
        return self.message(content, role="system")
    
    def user_message(self, content):
        """
        Creates a user message dictionary for the Groq model.

        :param content: The content of the user message.
        :return: A dictionary representing the user message.
        """
        return self.message(content, role="user")
    
    def assistant_message(self, content):
        """
        Creates an assistant message dictionary for the Groq model.

        :param content: The content of the assistant message.
        :return: A dictionary representing the assistant message.
        """
        return self.message(content, role="assistant")
    
    def __add_message(self, message):
        """
        Adds a message to the list of messages for the Groq model.

        :param message: The message dictionary to be added.
        :return: The added message dictionary.
        """
        if isinstance(message, dict):
            self.messages.append(message)
        else:
            raise ValueError("Message must be a dictionary.")
        
        return message

    def generate_response(self):
        """
        Generates a response from the Groq model based on the provided prompt.

        :return: The assistant's response as a dictionary.
        """
        if not self.messages:
            raise ValueError("No messages to send. Please add a message before generating a response.")

        payload = {
            "model": self.model_name,
            "messages": self.messages,
        }
        
        response = requests.post(self.url, headers=self.headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code}, {response.text}")
        
        parsed_response = self.parse_response(response.json())
        return self.assistant_message(parsed_response)
    
    def parse_response(self, response):
        """
        Parses the response from the Groq model.

        :param response: The JSON response from the model.
        :return: The content of the assistant's message.
        """
        if 'choices' in response and len(response['choices']) > 0:
            return response['choices'][0]['message']['content']
        else:
            raise ValueError("Invalid response format.")

    def reset(self):
        """
        Resets the messages list for the Groq model.
        """
        self.messages = []
        return self.messages