import streamlit as st
import os
import google.generativeai as genai


st.title("Chat - Talk2George")


# Set Google API key
os.environ['GOOGLE_API_KEY'] = "AIzaSyBEW2ADL9Z8DSAprqzBwxIz1tDBAqGO17c"
genai.configure(api_key = os.environ['GOOGLE_API_KEY'])


# Create the Model
model = genai.GenerativeModel('gemini-pro')


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role":"assistant",
            "content":"Ask me Anything"
        }
    ]


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Process and store Query and Response
def llm_function(query):
    response = model.generate_content(query)


    # Displaying the Assistant Message
    with st.chat_message("assistant"):
        st.markdown(response.text)


    # Storing the User Message
    st.session_state.messages.append(
        {
            "role":"user",
            "content": query
        }
    )


    # Storing the User Message
    st.session_state.messages.append(
        {
            "role":"assistant",
            "content": response.text
        }
    )


   
# Accept user input
query = st.chat_input("What is up?")


# Calling the Function when Input is Provided
if query:
    # Displaying the User Message
    with st.chat_message("user"):
        st.markdown(query)


    llm_function(query)
