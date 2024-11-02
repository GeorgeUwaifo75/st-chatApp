import streamlit as st

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




#Handle URL Input
def handle_urlinput(url_input):
    urls.append(url_input)
    st.write("URL Appended:",urls)
    
    
# Processing URLs
def get_web_text():
    text = ""
    loader = UnstructuredURLLoader(

    urls     
    #urls = [
    #    "https://crypto.news/tag/meme-coin/",
    #    "https://bravenewcoin.com/insights/cardano-and-ethereum-whales-betting-big-on-hot-new-popular-meme-coin-cutoshi-after-its-listing-on-cmc"
    #    ] 
    )
    data = loader.load()

    #text = data[0].page_content
    for i in range(len(data)):
        text += data[i].page_content

    return text


# Splitting text into small chunks to create embeddings
def get_text_chunks(text):
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
    urls = []
    
    st.set_page_config(page_title="Chat with multiple pdfs", page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple URLs :links:")

    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        #pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        

        url_input = st.text_input("Source URL:")

        if url_input:
            handle_urlinput(url_input)

        if st.button("Process"):
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
