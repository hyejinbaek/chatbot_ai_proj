import os
from dotenv import load_dotenv

import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# 임베딩
embeddings = OpenAIEmbeddings()

# 임베딩 차원 크기를 계산
dimension_size = len(embeddings.embed_query("hello world"))
print(dimension_size)

# FAISS 벡터 저장소 생성
db = FAISS(
    embedding_function=OpenAIEmbeddings(),
    index=faiss.IndexFlatL2(dimension_size),
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# 파일 읽기 및 Document 객체 변환
file_path = '../dataset/data_sample_2.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    file_content = file.read()

# 텍스트를 Document 객체로 변환
documents = [Document(page_content=file_content)]

# FAISS에 문서 추가
db = FAISS.from_documents(documents=documents, embedding=OpenAIEmbeddings())
print(db.index_to_docstore_id)
print(db.docstore._dict)
