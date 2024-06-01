import os
import pickle
import streamlit as st
from haystack.nodes import DensePassageRetriever
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline

st.title("DPR on Supreme Court Judgements (Capital Gain)")

# Get the current directory of the script
current_dir = os.path.dirname(__file__)

# Construct the full path to the file
file_path = os.path.join(current_dir, "inmemory_document_store.pkl")

# Open the file
with open(file_path, "rb") as f:
    document_store = pickle.load(f)

retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
)

reader = FARMReader(model_name_or_path="deepset/bert-base-cased-squad2")

pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever)

query = st.text_input("Enter your query:", "")

if query:
    with st.spinner("Searching..."):
        results = pipeline.run(query=query, params={"Retriever": {"top_k": 5}})
        for answer in results['answers']:
            st.markdown(f"=====================\nAnswer: {answer.answer}\nContext: {answer.context}\nScore: {answer.score}")
