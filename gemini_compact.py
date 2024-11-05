import streamlit as st
import pandas as pd

from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredURLLoader

from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

urls = []


#Handle URL Input
def handle_urlinput(url_input):
    urls.append(url_input)

def get_web_text():
    
    text = ""
    #main_placeholder.text("Processing...")
    loader = UnstructuredURLLoader(

    urls     
    )
    data = loader.load()
    
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

    
    st.sidebar .title("Source of Doc.")
    doc_type = st.sidebar.selectbox("Pick Doc Source", ("URL", "PDF", "Text"))

    if doc_type == "URL":
        st.sidebar.write("URL")
    elif doc_type == "PDF":
        st.sidebar.write("PDF")
    else:
        st.sidebar.write("Text")
    
    
    

if __name__ == '__main__':
    main()
