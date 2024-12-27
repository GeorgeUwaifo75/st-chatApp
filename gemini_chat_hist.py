import streamlit as st
import openai

# Load the OpenAI API key from the secrets file
openai.api_key = st.secrets["OPENAI_API_KEY"]

def chatbot_response(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0].text.strip()

def main():
    st.title("ChatGPT-like App")

    # Initialize the chat history
    chat_history = []

    # Create a container for the chat history
    chat_container = st.beta_container()

    # Create a function to render the chat history
    def render_chat_history():
        with chat_container:
            for message in chat_history:
                if message.startswith("User:"):
                    st.markdown(f"<div style='text-align: left;'><img src='https://i.imgur.com/WK7ZwzQ.png' width='30' height='30'><p style='font-size: 16px;'>{message}</p></div>")
                else:
                    st.markdown(f"<div style='text-align: right;'><img src='https://i.imgur.com/CHX5JvY.png' width='30' height='30'><p style='font-
