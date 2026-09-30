"""
비상용 대체 버전 — 원본(05_vectorstore.py)의 OpenAIEmbeddings를 로컬 sLLM 임베딩으로 교체.
FAISS는 어떤 임베딩 모델을 쓰든 상관없이 동일하게 동작한다 — 바뀐 건 embeddings 줄 하나뿐이다.
"""

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# 1. 문서 로드
loader = TextLoader("data/sample.txt", encoding="utf-8")
docs = loader.load()

# 2. Chunk 분할
splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,
    chunk_overlap=20,
    add_start_index=True,
)
chunks = splitter.split_documents(docs)

# 3. Embedding 모델 — 로컬 sLLM 임베딩으로 교체
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 4. FAISS Vector Store 생성
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings,
)

# 5. 유사한 문서 검색
query = "책은 며칠 동안 빌릴 수 있나요?"
results = vectorstore.similarity_search(query, k=3)

for i, doc in enumerate(results, start=1):
    print(f"\n[{i}]")
    print(doc.page_content)
    print(doc.metadata)
