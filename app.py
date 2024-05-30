import json
import streamlit as st
# from haystack.document_stores import FAISSDocumentStore
from haystack.document_stores import InMemoryDocumentStore
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from haystack.nodes import DensePassageRetriever
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline

st.header("DPR on Supreme Court Judgements (Capital Gain)")

with open("responses.json", 'r') as f:
  data = json.load(f)

documents = [
    {
        "content": doc["text"],
        "meta": {
            "name": doc["title"],
            "url": doc["url"]
        }
    } for doc in data
]

# document_store = FAISSDocumentStore(embedding_dim=768, faiss_index_factory_str="Flat")
document_store = InMemoryDocumentStore()
document_store.write_documents(documents)

retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
)

document_store.update_embeddings(retriever)
# document_store.save("faiss_index")

reader = FARMReader(model_name_or_path="deepset/bert-base-cased-squad2")

pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever)

query = st.input()
# query = "What is the subject matter of the petition in the Sadanand S. Varde case?"
result = pipeline.run(query=query, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}})

for answer in result['answers']:
    st.markdown(f"=====================\nAnswer: {answer.answer}\nContext: {answer.context}\nScore: {answer.score}")