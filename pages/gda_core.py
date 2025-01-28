import os
import time
import streamlit as st
import pandas as pd
import requests
import json
import re
import urllib.request

from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredURLLoader

from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain_community.llms import HuggingFaceEndpoint


json_url = 'https://api.npoint.io/03cc552f40aca75a2bf1'
#json_url = os.environ.get("JSON_URL")
response = requests.get(json_url)
json_data = response.content

urls = []


#def clear_text():
#    st.session_state.my_text = st.session_state.widget
#    st.session_state.widget = ""


#Upload IvieAI dataset
def upload_ivieAi():
    # Load the JSON data into a Python dictionary
    data = json.loads(json_data)

    # Extract the first "reply" values from each item in "allpushdata"
    text = ""
    for item in data["allpushdata"]:
        first_reply = item["replies"][0]["reply"]
        #replies.append(first_reply)
        text += first_reply + "\n"
    
    return text

#Handle URL Input
def handle_urlinput(url_input):
    urls.append(url_input)

# Processing pdfs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
    
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
    #embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def generate_answer(question):
    response = st.session_state.conversation({"question": question})

    st.session_state.chat_history = response['chat_history']
    #st.write(response)  # Return only the answer from the response

    answer = response.get("answer").split("Helpful Answer:")[-1].strip()
    explanation = response.get("source_documents", [])
    doc_source = [d.page_content for d in explanation]
    
    #st.write("Source:",doc_source)
    #New  
    #display_chat_history()
    return answer, doc_source, response

def display_chat_history():
    """
    Displays the chat history from the session state in a readable format.
    """
    #st.header("GiTeksol :green[Document] Assistant [*:blue[GDA]*]")

    if st.session_state.chat_history:
        st.write(":green[*Chat History:*]")
        for i, message in enumerate(st.session_state.chat_history):
            if hasattr(message, "content"):
               content = message.content
            elif hasattr(message, "text"):
                 content = message.text
            else:
                 content = str(message)
            
            if "HumanMessage" in str(message):
                st.write(f"  Human {i//2 + 1}: {content}")
            elif "AIMessage" in str(message):
                st.write(f"  AI {i//2 + 1}: {content}")
            else:
                #st.write(f"  Unrecognized Message {i//2 +1}: {content}")
                if i%2 == 0:
                    with st.chat_message("user"):
                        st.markdown(content)
                else:
                    with st.chat_message("assistant"):
                        st.markdown(content)
                
    else:
        st.write("No chat history yet.")

# Handling user questions 
def handle_userinput(question):
        
    # Add user question
    #with st.chat_message("user"):
    #    st.markdown(question)

    # Answer the question
    answer, doc_source, response = generate_answer(question)
   
    with st.chat_message("assistant"):
        st.write(answer)
        #st.markdown(f"<p style='color:brown;'>{answer}</p>", unsafe_allow_html=True) 
        
    #st.write(response)
    if st.session_state.chat_history_displayed == True:
        display_chat_history()
    st.session_state.chat_history_displayed = True


def handle_userinput2(question):
    
    # Answer the question
    answer, doc_source, response = generate_answer(question)
   
    with st.chat_message("assistant"):
        # Split the answer into lines
        lines = answer.splitlines()

        for line in lines:
            # Split each line into words
            words = line.split()
            
            for word in words:
                st.write(word, end=" ", flush=True)
                time.sleep(0.05)  # Adjust delay as needed
            
            # Add a newline after each line
            st.write("")

       
    if st.session_state.chat_history_displayed == True:
        display_chat_history()
    st.session_state.chat_history_displayed = True

# Storing converstations as chain of outputs
def get_conversation_chain(vectorstore):
    #llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro')

   
    llm = HuggingFaceEndpoint(
    #endpoint_url="mistralai/Mistral-7B-Instruct-v0.3/",temperature=0.25, max_length=512)
    endpoint_url="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",temperature=0.25, max_length=512)
    
    #endpoint_url="mistralai/Mistral-7B-Instruct-v0.3/",temperature=0.12, max_length=512)  ...The last 
    #endpoint_url="mistralai/Mistral-7B-Instruct-v0.2/",temperature=0.65, max_length=512)
    #endpoint_url="Qwen/QwQ-32B-Preview",temperature=0.65, max_length=512)
    #endpoint_url="meta-llama/Llama-3.1-8B-Instruct/",temperature=0.3, max_length=512)

    st.write("[Mistral-V0.3]")
    #st.write("[Gemini")
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def get_conversation_chain2(vectorstore):
    llm = HuggingFaceEndpoint(
    endpoint_url="mistralai/Mistral-7B-Instruct-v0.2/",temperature=0.1, max_length=512)
    conversation_chain = load_qa_chain(llm, chain_type="stuff")
    return conversation_chain

def main():
    load_dotenv()
    
    st.set_page_config(page_title="GiTeksol Document Assistant", page_icon=":books:")
    #st.write("This is :blue[test]")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
   
    # Initialize the flag
    if "chat_history_displayed" not in st.session_state:
        st.session_state.chat_history_displayed = False
        
    st.header("GiTeksol :green[Document] Assistant [*:blue[GDA]*]")

    #display_chat_history()
    user_question = st.text_input("Ask a question about your documents:")
    #user_question = st.text_input("Ask a question about your documents:", key='widget', on_change=clear_text)  
   
    # Ask a question
    if user_question:
        handle_userinput(user_question)
       
        
    
    #DCD6D0    
    
    st.markdown("""
    <style>
        [data-testid=stSidebar] {
            background-color:  #87CEFA;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.sidebar.title(":red[Source of Doc.]")
    doc_type = st.sidebar.selectbox("Pick Doc Source", ("Doc Types","URL", "PDF", "IvieAI"))

    if doc_type == "URL" or "PDF":
        with st.sidebar:
                st.subheader("Doc Sources...")
                if doc_type == "URL": 
                    for i in range(3):
                        url_input = st.sidebar.text_input(f"Source URL{i+1}:")
                        handle_urlinput(url_input)
                elif doc_type == "PDF":
                       pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
                
                process_url = st.button("Process Docs")
                if process_url:    
                    with st.spinner("Processing"):
                            
                            if doc_type == "URL":
                                raw_text = get_web_text()
                            elif doc_type == "PDF":
                                raw_text = get_pdf_text(pdf_docs)
                            elif doc_type == "IvieAI":
                                raw_text = upload_ivieAi()
                                
                            #convert to chunks
                            text_chunks = get_text_chunks(raw_text)
                            #st.write(text_chunks)

                            #embeddings
                            vectorstore = get_vectorstore(text_chunks)

                            #create conversation chain
                            st.session_state.conversation = get_conversation_chain(vectorstore)
                            #st.session_state.conversation = get_conversation_chain2(vectorstore)
    
    elif doc_type == "PDF":
        st.sidebar.write("PDF")
    else:
        st.sidebar.write("IvieAI")

if __name__ == '__main__':
    main()
