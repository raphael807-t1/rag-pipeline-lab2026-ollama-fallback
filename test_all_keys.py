import os
import requests
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("🔑 RAG 파이프라인 통합 API Key 상태 검증 스크립트")
print("=" * 60)

# 1. Gemini (Google) API Key
gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
print("\n[1] Gemini (Google) API Key 진단")
if gemini_key:
    r = requests.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={gemini_key}")
    if r.status_code == 200:
        print("  ✅ 결과: 정상 작동! (Gemini API 호출 가능)")
    else:
        print(f"  ❌ 결과: 오류 발생 (상태코드: {r.status_code}) -> {r.text[:200]}")
else:
    print("  ⚠️ 결과: .env에 GEMINI_API_KEY (또는 GOOGLE_API_KEY)가 설정되어 있지 않습니다.")

# 2. Upstage API Key
upstage_key = os.getenv("UPSTAGE_API_KEY")
print("\n[2] Upstage (Solar) API Key 진단")
if upstage_key:
    r = requests.post(
        "https://api.upstage.ai/v1/solar/chat/completions",
        headers={"Authorization": f"Bearer {upstage_key}", "Content-Type": "application/json"},
        json={"model": "solar-mini", "messages": [{"role": "user", "content": "hi"}]},
    )
    if r.status_code == 200:
        print("  ✅ 결과: 100% 정상 작동! (Solar-Mini 답변 가능)")
    else:
        print(f"  ❌ 결과: 오류 발생 (상태코드: {r.status_code}) -> {r.text[:200]}")
else:
    print("  ⚠️ 결과: .env에 UPSTAGE_API_KEY가 없습니다.")

# 3. Pinecone API Key
pinecone_key = os.getenv("PINECONE_API_KEY")
print("\n[3] Pinecone Vector DB API Key 진단")
if pinecone_key:
    r = requests.get("https://api.pinecone.io/indexes", headers={"Api-Key": pinecone_key})
    if r.status_code == 200:
        print("  ✅ 결과: 100% 정상 작동! (Vector DB 연동 가능)")
    else:
        print(f"  ❌ 결과: 오류 발생 (상태코드: {r.status_code}) -> {r.text[:200]}")
else:
    print("  ⚠️ 결과: .env에 PINECONE_API_KEY가 없습니다.")

# 4. LangSmith API Key
langsmith_key = os.getenv("LANGSMITH_API_KEY")
print("\n[4] LangSmith Tracing API Key 진단")
if langsmith_key:
    r = requests.get("https://api.smith.langchain.com/info", headers={"x-api-key": langsmith_key})
    if r.status_code == 200:
        print("  ✅ 결과: 100% 정상 작동! (트레이싱 로그 수집 가능)")
    else:
        print(f"  ❌ 결과: 오류 발생 (상태코드: {r.status_code}) -> {r.text[:200]}")
else:
    print("  ⚠️ 결과: .env에 LANGSMITH_API_KEY가 없습니다.")

print("\n" + "=" * 60)
