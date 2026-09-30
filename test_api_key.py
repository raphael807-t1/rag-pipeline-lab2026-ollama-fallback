"""
OpenAI API Key가 유효한지, 유효하다면 잔액(크레딧)이 남아있는지 확인하는 스크립트.

두 단계로 나눠서 확인한다:
  1. /v1/models 호출 — 요금이 들지 않는 조회성 API. 키 자체가 맞는지만 검증한다.
     -> 401(invalid_api_key)이면 키 자체가 틀렸다는 뜻.
  2. /v1/embeddings 호출 — 실제로 "사용"하는 API. 아주 짧은 텍스트라 비용은 사실상 0에 수렴.
     -> 429(insufficient_quota)면 키는 맞지만 잔액이 없다는 뜻.
     -> 200이면 완전히 정상.
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("❌ .env에 OPENAI_API_KEY가 설정되어 있지 않습니다.")
    raise SystemExit(1)

print(f"[정보] 키 앞 12자리: {api_key[:12]}... (전체 길이: {len(api_key)}자)")
print()

headers = {"Authorization": f"Bearer {api_key}"}

# 1단계: 키 자체가 유효한지 (요금 발생 없음)
print("=== 1단계: 키 인증 확인 (/v1/models, 무료) ===")
resp = requests.get("https://api.openai.com/v1/models", headers=headers)
print(f"HTTP 상태코드: {resp.status_code}")

if resp.status_code == 401:
    print("❌ 결과: 키 자체가 틀렸습니다 (invalid_api_key).")
    print("   -> 복사/붙여넣기 과정에서 오타가 있었거나, 이미 폐기(rotate)된 키일 수 있습니다.")
    raise SystemExit(1)
elif resp.status_code != 200:
    print(f"⚠️ 예상 못한 상태코드입니다. 응답: {resp.text[:300]}")
    raise SystemExit(1)

print("✅ 키 인증 자체는 정상입니다.")
print()

# 2단계: 실제 사용(요금 발생) 가능한지, 즉 잔액이 있는지
print("=== 2단계: 잔액 확인 (/v1/embeddings, 극소 비용) ===")
resp2 = requests.post(
    "https://api.openai.com/v1/embeddings",
    headers={**headers, "Content-Type": "application/json"},
    json={"model": "text-embedding-3-small", "input": "test"},
)
print(f"HTTP 상태코드: {resp2.status_code}")

if resp2.status_code == 429:
    body = resp2.json()
    code = body.get("error", {}).get("code", "")
    if code == "insufficient_quota" or "credit" in body.get("error", {}).get("message", "").lower():
        print("❌ 결과: 키는 유효하지만 계정 잔액(크레딧)이 없습니다 (insufficient_quota).")
        print("   -> 이 키를 발급한 쪽(회사/담당자)에게 충전 또는 재발급을 요청해야 합니다.")
    else:
        print(f"⚠️ 429지만 다른 이유입니다: {body}")
    raise SystemExit(1)
elif resp2.status_code == 200:
    print("✅ 결과: 키가 완전히 정상 작동합니다 (인증 + 잔액 모두 OK).")
    print()

    # 3단계: 진짜 성공했는지 눈으로 보이게, 간단한 채팅 질문 하나 던져본다.
    print("=== 3단계: 실제 응답 확인 (gpt-4o-mini에게 질문) ===")
    resp3 = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={**headers, "Content-Type": "application/json"},
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "오늘 날씨 어때?"}],
        },
    )
    if resp3.status_code == 200:
        answer = resp3.json()["choices"][0]["message"]["content"]
        print("질문: 오늘 날씨 어때?")
        print(f"답변: {answer}")
    else:
        print(f"⚠️ 채팅 호출 실패. 상태코드: {resp3.status_code}, 응답: {resp3.text[:300]}")
else:
    print(f"⚠️ 예상 못한 상태코드입니다. 응답: {resp2.text[:300]}")
