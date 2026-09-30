import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 1. .env 파일의 OPENAI_API_KEY 로드
load_dotenv()

# 2. AI 모델 객체 생성 (gpt-4o-mini 사용)
llm = ChatOpenAI(model="gpt-4o-mini")

# 3. 질문 던지고 답변 출력하기
response = llm.invoke("오늘 날씨 어때?")

print("답변:", response.content)