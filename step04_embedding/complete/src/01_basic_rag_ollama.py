"""
비상용 대체 버전 — OpenAI API가 막혔을 때 로컬 sLLM(Ollama)으로 같은 RAG 흐름을 체험한다.

원본(01_basic_rag.py)과 로직은 완전히 동일하다. 바뀐 건 딱 두 줄뿐이다:
  - OpenAIEmbeddings  -> OllamaEmbeddings (nomic-embed-text, 로컬 임베딩 모델)
  - ChatOpenAI        -> ChatOllama (qwen2.5:3b, 로컬 sLLM)
API Key도, 인터넷 연결도, 과금도 필요 없다 — 전부 내 컴퓨터 안에서 돈다.
"""

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()

# 1. 검색 대상으로 사용할 아주 작은 문서 집합 (원본과 동일)
pages = [
    Document(page_content="샘플도서관 평일 운영시간은 오전 9시부터 오후 9시까지이다."),
    Document(page_content="도서는 최대 5권까지 14일 동안 대출할 수 있다."),
    Document(page_content="노트북 이용 공간은 2층 디지털자료실이다."),
]

# 2. 문장의 의미를 벡터로 변환할 모델 — 로컬 임베딩 모델로 교체
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 3. 메모리 기반 Vector Store 생성
vectorstore = InMemoryVectorStore.from_documents(
    pages,
    embedding=embeddings,
)

# 4. Retriever 생성 (k=2: 관련 문서 상위 2개 반환)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

question = "도서관에서 노트북은 어디에서 사용할 수 있나요?"

# 5. 질문과 관련된 문서 검색
retrieved_docs = retriever.invoke(question)

print("[검색 결과]")
for i, doc in enumerate(retrieved_docs, start=1):
    print(f"{i}. {doc.page_content}")

# 6. 검색 결과를 하나의 Context 문자열로 변환
context = "\n\n".join(doc.page_content for doc in retrieved_docs)

# 7. Prompt 구성 (원본과 동일)
prompt = ChatPromptTemplate.from_template("""
당신은 제공된 문서를 근거로 답하는 질문-답변 도우미입니다.
문서에 없는 내용은 추측하지 말고 "제공된 문서에서 확인할 수 없습니다."라고 답하세요.

[문서]
{context}

[질문]
{question}

[답변]
""")

# 8. LLM 생성 — 로컬 sLLM으로 교체 (qwen2.5:3b)
llm = ChatOllama(model="qwen2.5:3b", temperature=0)

messages = prompt.invoke({"context": context, "question": question})
response = llm.invoke(messages)

print("\n[최종 답변]")
print(response.content)
