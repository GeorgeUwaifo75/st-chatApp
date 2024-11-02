import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter

st.title("URL load and Analyzer")
loader = UnstructuredURLLoader(
    urls = [
        "https://www.moneycontrol.com/news/india/two-women-die-6-taken-ill-after-consuming-mango-kernel-gruel-in-odisha-12856689.html",
        "https://www.nimbleway.com/blog/use-serp-api-to-boost-rankings-and-explore-markets"

    ] 
)
data = loader.load()

text = data[0]
text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
   

st.write(data[0].metadata)
