import streamlit as st
from transformers import pipeline

# Load the model
model = pipeline("text-generation", model="t5-base")

# Set a default model if "model" not in st.session_state:
st.session_state["model"] = model

# Initialize chat history if "messages" not in st.session_state:
st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("What is up?"):
# Add user message to chat history
st.session_state.messages.append({"role": "user", "content": prompt})

# Generate a response from the model
response = st.session_state["model"](prompt, max_length=50, do_sample=True)

# Display the response in a chat message container
with st.chat_message("assistant"):
    st.markdown(response[0]["generated_text"])
