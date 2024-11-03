import streamlit as st
import urllib.request
import pandas as pd

from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredURLLoader

from dotenv import load_dotenv
from PyPDF2 import PdfReader
#from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


urls = []

#Upload IvieAI dataset
def upload_ivieAi():
    json_url = 'https://api.npoint.io/03cc552f40aca75a2bf1'
    response = requests.get(json_url)
    json_data = response.content
   
    # Load the JSON data into a Python dictionary
    data = json.loads(json_data)

    # Extract the first "reply" values from each item in "allpushdata"
    text = ""
    replies = []
    for item in data["allpushdata"]:
        first_reply = item["replies"][0]["reply"]
        replies.append(first_reply)
        text += first_reply + "\n"

#Handle URL Input
def handle_urlinput(url_input):
    urls.append(url_input)
    #st.text("URL Appended.")
    
    
# Processing URLs
def get_web_text():
    text = ""
    #main_placeholder.text("Processing...")
    loader = UnstructuredURLLoader(

    urls     
    )
    data = loader.load()

    #text = data[0].page_content
    for i in range(len(data)):
        text += data[i].page_content

    return text


# Splitting text into small chunks to create embeddings
def get_text_chunks(text):
   # main_placeholder.text("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ["\n\n", "\n", " "],
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# Using Google's embedding004 model to create embeddings and FAISS to store the embeddings
def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


# Handling user questions
def handle_userinput(question):
    response = st.session_state.conversation({"question": question})
    st.session_state.chat_history = response['chat_history']
    st.write(response)  # Return only the answer from the response

# Storing converstations as chain of outputs
def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro-latest')
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain



def main():
    load_dotenv()
    
    st.set_page_config(page_title="Chat with multiple URLs", page_icon=":books:")
    
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Research with multiple URLs :links:")

    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("URL Sources...")

        for i in range(3):
            url_input = st.sidebar.text_input(f"Source URL{i+1}:")
            handle_urlinput(url_input)
            
        
        if st.button("Load IvieAI"):
            upload_ivieAi()

        
        process_url = st.button("Process URL(s)")
        
        if process_url:    
            with st.spinner("Processing"):
                raw_text = get_web_text()

                #convert to chunks
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)

                #embeddings
                vectorstore = get_vectorstore(text_chunks)


                #create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
    

if __name__ == '__main__':
    main()
