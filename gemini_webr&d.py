import streamlit as st
import pandas as pd
import numpy as np
import faiss
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredURLLoader
#from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

st.title("URL load and Analyzer")
loader = UnstructuredURLLoader(
    urls = [
        "https://www.moneycontrol.com/news/india/two-women-die-6-taken-ill-after-consuming-mango-kernel-gruel-in-odisha-12856689.html",
        "https://www.nimbleway.com/blog/use-serp-api-to-boost-rankings-and-explore-markets"

    ] 
)
data = loader.load()

text = data[0].page_content

#st.write(text)

#text_splitter = CharacterTextSplitter(
text_splitter = RecursiveCharacterTextSplitter(  
# The Separator below is for the Recursive format   
        separators = ["\n\n", "\n", " "],
        #separator = ["\n"],
        chunk_size = 200,
        chunk_overlap = 0,
        length_function = len
    )
chunks = text_splitter.split_text(text)
   
st.write(len(chunks))
#for chunk in chunks:
#    st.write(len(chunk))
pd.set_option('display.max_colwidth',100)
df = pd.read_csv("salaries.csv")
st.write(df.shape)

st.text(df.head())

encoder = SentenceTransformer("all-mpnet-base-v2") 
vectors = encoder.encode(df.job_title)
 
dim = vectors.shape[1]
st.write(dim)

index = faiss.IndexFlatL2(dim)
index.add(vectors)

search_query = "How many Data Analyst jobs exist?"
vec = encoder.encode(search_query)
st.write("Vec:",vec.shape)


svec = np.array(vec).reshape(1,-1)
st.write("Svec:",svec.shape)

distances, I = index.search(svec, k=2)

st.write("I:", I) 
#st.write("I val:",I)

#st.write("Loc value is:", df.iloc[I[0]])


#st.write("Loc value:", df[I[0]])
