"""
비상용 대체 버전 — 원본(04_embedding.py)의 OpenAIEmbeddings를 로컬 sLLM 임베딩으로 교체.
바뀐 건 embeddings 생성 줄 하나뿐이다.
"""

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings

load_dotenv()

embeddings = OllamaEmbeddings(model="nomic-embed-text")

text = "도서는 최대 5권까지 대출할 수 있다."
vector = embeddings.embed_query(text)

print("벡터 길이:", len(vector))
print("앞부분 10개 값:")
print(vector[:10])
