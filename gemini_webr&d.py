import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredURLLoader

st.title("URL load and Analyzer")
loader = UnstructuredURLLoader(
    urls = [
        "https://www.moneycontrol.com/news/india/two-women-die-6-taken-ill-after-consuming-mango-kernel-gruel-in-odisha-12856689.html",
        "https://www.nimbleway.com/blog/use-serp-api-to-boost-rankings-and-explore-markets"

    ] 
)
data = loader.load()
st.write("The number of URLs: ",len(data))
