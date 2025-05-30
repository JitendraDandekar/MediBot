import os
import torch
import streamlit as st
from dotenv import load_dotenv
from medibot import MediBot

# Load environment variables
load_dotenv()

# Ensure the torch classes path is set correctly
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

# Initialize MediBot
medibot = MediBot()

# Page configuration
st.set_page_config(page_title="MediBot", page_icon="🤖", layout="wide")
st.markdown("<h1 style='text-align: center;'>🤖 MediBot</h1>", unsafe_allow_html=True)

# Message container with fixed height
messages_container = st.container(height=600)
user_input_container = st.container()

# Initialize session messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am MediBot, your virtual assistant. How can I help you today?"}
    ]

# Display all messages in chat
with messages_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# User input at the bottom
if query := user_input_container.chat_input("Say something"):
    # Store user message
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Display user message
    with messages_container:
        with st.chat_message("user"):
            st.markdown(query)

    # Process the query and generate a response
    response = medibot.generate_response(query)

    # Simulated assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Display assistant response
    with messages_container:
        with st.chat_message("assistant"):
            st.markdown(response)
