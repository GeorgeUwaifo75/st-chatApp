import streamlit as st
import openai
import rag

# Load the OpenAI API key from the secrets file
openai.api_key = st.secrets["OPENAI_API_KEY"]

def chatbot_response(prompt, rag_model):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0].text.strip()

def rag_scorer(text):
    rag_model.load("rag-model")
    scores = rag_model.score(text)
    return scores["rationality"], scores["aggression"], scores["greed"]

def main():
    st.title("ChatGPT-like App with RAG")

    # Initialize the chat history
    chat_history = []

    # Create a container for the chat history
    chat_container = st.beta_container()

    # Create a function to render the chat history
    def render_chat_history():
        with chat_container:
            for message in chat_history:
                if message.startswith("User:"):
                    role = "user"
                else:
                    role = "assistant"
                st.markdown(f'<div class="chat-message {role}">{message}</div>')

    # Create a function
