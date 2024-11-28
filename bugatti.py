import os
import streamlit as st
import pandas as pd
import requests
import json
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


#Upload IvieAI dataset
def upload_ivieAi():
    # Load the JSON data into a Python dictionary
    data = json.loads(json_data)

    # Extract the first "reply" values from each item in "allpushdata"
    # replies = []
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
    st.write(response)  # Return only the answer from the response
   
    
    answer = response.get("answer").split("Helpful Answer:")[-1].strip()
    explanation = response.get("source_documents", [])
    doc_source = [d.page_content for d in explanation]

    return answer, doc_source



# Handling user questions 
def handle_userinput(question):
    #response = st.session_state.conversation({"question": question})
    #st.session_state.chat_history = response['chat_history']
    #st.write(response)  # Return only the answer from the response
   
 
# Append user question to history
    #st.session_state.chat_history.append({"role": "user", "content": question})
    
    # Display chats
    #for message in st.session_state.chat_history:
    #    with st.chat_message(message["role"]):
    #        st.markdown(message["content"])

    
    # Add user question
    with st.chat_message("user"):
        st.markdown(question)

    # Answer the question
    answer, doc_source = generate_answer(question)
   
    with st.chat_message("assistant"):
        st.write(answer)
    
  # Append assistant answer to history
    #st.session_state.chat_history.append({"role": "assistant", "content": answer})
    
    # Append the document sources
    #st.session_state.source.append({"question": question, "answer": answer, "document": doc_source})




# Storing converstations as chain of outputs
def get_conversation_chain(vectorstore):
    #llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro')

    llm = HuggingFaceEndpoint(
    endpoint_url="mistralai/Mistral-7B-Instruct-v0.2/",temperature=0.2, max_length=512)
    
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

   
    st.header("GiTeksol :green[Document] Assistant [*:blue[GDA]*]")

    user_question = st.text_input("Ask a question about your documents:")

   
    # Ask a question
    if user_question:
        handle_userinput(user_question)
    
    # #87CEFA    
    
    st.markdown("""
    <style>
        [data-testid=stSidebar] {
            background-color:  #DCD6D0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.sidebar.title(":green[Source of Doc.]")
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
                            st.write(text_chunks)

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
