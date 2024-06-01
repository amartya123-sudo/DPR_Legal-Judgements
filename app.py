import json
import pickle
import streamlit as st
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from haystack.nodes import DensePassageRetriever
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline

st.title("DPR on Supreme Court Judgements (Capital Gain)")

# with open("responses.json", 'r') as f:
#   data = json.load(f)

# documents = [
#     {
#         "content": doc["text"],
#         "meta": {
#             "name": doc["title"],
#             "url": doc["url"]
#         }
#     } for doc in data
# ]

# document_store = FAISSDocumentStore(embedding_dim=768, faiss_index_factory_str="Flat", sql_url="sqlite:///faiss_document_store.d")
with open("inmemory_document_store.pkl", "rb") as f:
    document_store = pickle.load(f)
# document_store.write_documents(documents)

# document_store = FAISSDocumentStore.load(index_path="./faiss_index", config_path="./faiss_index.json")

retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
)

# document_store.update_embeddings(retriever)
# document_store.save(index_path="./faiss_index", config_path="./faiss_index.json")
# with open("inmemory_document_store.pkl", "wb") as f:
#     pickle.dump(document_store, f)


reader = FARMReader(model_name_or_path="deepset/bert-base-cased-squad2")

pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever)

query = st.text_input("Enter your query:", "")

if query:
    with st.spinner("Searching..."):
        results = pipeline.run(query=query, params={"Retriever": {"top_k": 5}})
        for answer in results['answers']:
            st.markdown(f"=====================\nAnswer: {answer.answer}\nContext: {answer.context}\nScore: {answer.score}")

# query = st.text_input("Enter Question")
# query = "What is the subject matter of the petition in the Sadanand S. Varde case?"
# result = pipeline.run(query=query, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}})
# for answer in result['answers']:
#    print(f"=====================\nAnswer: {answer.answer}\nContext: {answer.context}\nScore: {answer.score}")