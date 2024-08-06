from pymilvus import MilvusClient, DataType
from pymilvus import connections, db
from pymilvus import Collection, CollectionSchema, FieldSchema, utility
from pymilvus import connections
from FlagEmbedding import BGEM3FlagModel
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np 
import faiss
import torch
import argparse
import faiss
import os

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# 1. 가상 문서 생성
def generate_hypothetical_document(query, model, tokenizer):
    inputs = tokenizer.encode(query, return_tensors='pt').to(device)
    outputs = model.generate(inputs, max_length=100)
    hypothetical_doc = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return hypothetical_doc

# 2. 임베딩 생성
def get_embeddings(texts, embedder):
    embeddings = embedder.encode(texts, convert_to_tensor=True)
    return embeddings

# 3. 유사도 검색
def search_similar_documents(query_embedding, doc_embeddings, k=4):
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    faiss.normalize_L2(doc_embeddings)
    index.add(doc_embeddings)
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, k)
    return indices[0]

# 설정
query = "Compare the families of Emma Stone and Ryan Gosling"
document_texts = ["Document 1 text...", "Document 2 text...", "Document 3 text...", "Document 4 text..."]

# 모델 로드
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
# tokenizer.to(device)
print(device)
model.to(device)

# 가상 문서 생성
hypothetical_doc = generate_hypothetical_document(query, model, tokenizer)
print("Hypothetical Document:", hypothetical_doc)

# 임베딩 생성
query_embedding = get_embeddings([query, hypothetical_doc], embedder)
doc_embeddings = get_embeddings(document_texts, embedder)

# 유사도 검색
similar_docs_indices = search_similar_documents(query_embedding[1].cpu().unsqueeze(0).numpy(), doc_embeddings.cpu().numpy())
print("Similar Documents Indices:", similar_docs_indices)

# 유사한 문서 출력
for idx in similar_docs_indices:
    print(document_texts[idx])