import os
import time
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ============================
# 1. 설정 값
# ============================

INPUT_CSV_PATH = "/workspace/data/golden_11900_stratified_samples.csv"
OUTPUT_JSONL_PATH = "/workspace/data/cot_11900_dataset.jsonl"

TEACHER_MODEL = "gpt-5-mini"  # teacher LLM 이름
MAX_WORKERS = 5               # 병렬 쓰레드 수
MAX_RETRIES = 3               # 재시도 횟수
RETRY_BACKOFF_SECONDS = 10    # 재시도 사이 대기시간(초)

# OPENAI_API_KEY는 환경변수로만 관리 (코드에 하드코딩 금지)
if "OPENAI_API_KEY" not in os.environ:
    print("오류: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    exit(1)

client = OpenAI()

# ============================
# 2. Teacher용 SYSTEM PROMPT
# ============================

SYSTEM_PROMPT = """
당신은 한국어 **텍스트 역정규화(Inverse Text Normalization, ITN)**를 전문으로 하는 AI assistant입니다.
당신의 임무는 [입력 문장]을 [한국어 숫자 발화 및 패턴 지침]에 따라 분석하고,
그 과정을 [출력 형식]에 맞춰 정확하게 서술한 뒤,
마지막에 제공된 정답 scriptITN과 완전히 동일한 최종 변환 문장을 생성하는 것입니다.

이 출력은 **CoT(Chain-of-Thought) 파인튜닝을 위한 학습 데이터**로 사용됩니다.

아래의 규칙과 출력 형식을 반드시 준수하십시오.
임의로 내용을 추가하거나 가정하거나 추론하지 마십시오.
scriptTN에 존재하지 않는 정보는 절대로 만들어내지 마십시오.

## [한국어 숫자 발화 및 패턴 지침]

### 1. 월 범위 표현 ("일 에서 구 월" -> "1~9월")
   - {한글 숫자} + "에서"/"부터" + {한글 숫자} + "월" 구조는 "~"로 변환.
   - 예: "일 에서 구 월만 놓고 봤을 때" → "1~9월만 놓고 봤을 때"

### 2. 숫자 발화 사이의 '에', '다시' 패턴 (전화번호/카드번호)
   - 순수 숫자 발화 사이의 '에' → 구분 기호("-", 공백)로 변환.
   - '다시'는 반복 의미이며 하이픈 또는 동일 숫자 결합으로 처리 가능.
   - 예: "공 일 오 에 하나 둘 셋 넷 에 구 구 구 구" → "015-1234-9999"

### 3. 한국어 숫자 발화 패턴
   - "공" → 0
   - "하나, 둘, 셋…" → 1, 2, 3…
   - "열둘, 열여덟" 등 두 자리 수는 숫자로 변환.
   - "스물, 서른…" 등은 십 단위 표현.

### 4. 금액 및 큰 숫자
   - "십 이만 천 백 육십 구 명" → "12만 1,169명"
   - "구억 삼천 이백만 원" → "9억 3,200만 원"
   - "육백 점 팔 퍼센트" → "600.8%"

### 5. 차량 번호 (숫자+한글 혼합)
   - 숫자는 아라비아 숫자로 변환하고, 한글(후, 바, 거 등)은 그대로 유지.
   - 예: "[일 삼 후 오 영 사 일 번]" → "13후 5041번"

### 6. 제품 번호 (알파벳+숫자 혼합)
   - 알파벳 발음 + 숫자 발화 혼합은 제품/모델 번호.
   - "다시"는 하이픈(-)으로 표현 가능.
   - 예: "[에스 엠 에이 다시 비 엘 이 공 공 공]" → "SMA-BL2000"

### 7. 예외: 변환하지 않는 표현
   - "첫 번째", "스무 번째"
   - "한두 번", "대여섯 명"
   - 그대로 유지하며 변환하지 않음.
   - 단, "한 개", "두 가지" 등은 변환함. (→ 1개, 2가지)

## [출력 형식 – 반드시 준수]

출력은 **반드시 아래 3단계를 순서대로 포함**해야 합니다.
번호를 바꾸거나 생략하거나 재정렬하지 마십시오.
1. 분석, 2. 변환, 3. 최종 문장은 **최소 2문장, 최대 3문장 이하로 작성**하십시오.

### 1. 분석:
   - 입력 문장에서 숫자/패턴/단위 표현을 모두 식별.
   - 각 표현이 어떤 한국어 발화 특성을 가지는지 설명.
   - 예외 표현이 있다면 그대로 유지해야 하는 근거 설명.
   - (주의) 입력에 없는 정보는 만들지 말 것.

### 2. 변환:
   - 식별된 각 표현을 지침의 몇 번 규칙에 따라 어떻게 변환하는지 단계적으로 설명.
   - 예외 표현은 “변환하지 않는다”고 이유와 함께 명시.
   - hallucination 금지: 존재하지 않는 표현, 숫자, 단어 추가 금지.

### 3. 최종 문장:
   - 제공된 scriptITN 정답과 완전히 동일한 문장을 중괄호 { } 안에 한 번만 제시.
   - 절대로 새로운 변환 결과를 생성하지 말 것.
   - 예: {12만 1,169명 중 9만 3,939명이 남성이었다.}
"""

# ============================
# 3. 유틸 함수
# ============================

def extract_final_sentence(cot_output: str):
    """
    teacher가 생성한 CoT 텍스트에서
    3. 최종 문장: { ... } 부분만 파싱해서 반환.
    형식이 안 맞으면 None 반환.
    """
    m = re.search(r"3\.\s*최종 문장\s*:\s*\{(.+?)\}", cot_output, re.DOTALL)
    if not m:
        return None
    return m.group(1).strip()


def generate_cot_label(script_tn: str, script_itn: str, script_uuid: str):
    """
    단일 샘플에 대해 teacher LLM을 호출하여
    CoT 형식의 레이블을 생성.
    - script_tn: 입력 TN
    - script_itn: 정답 ITN
    - script_uuid: 샘플 식별자
    """
    for i in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=TEACHER_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"--- [입력 문장]:\n{script_tn}\n\n"
                            f"--- [정답 scriptITN]:\n{script_itn}\n\n"
                            "위의 입력 문장과 정답 scriptITN을 기준으로, "
                            "지침과 출력 형식에 맞는 CoT(분석/변환/최종 문장)를 작성하십시오."
                        )
                    }
                ],
                temperature=1.0,
                max_completion_tokens=4096,
            )

            cot_output = completion.choices[0].message.content.strip()
            print(f"DEBUG: UUID={script_uuid} CoT Output:\n{cot_output}\n---\n")
            final_sentence = extract_final_sentence(cot_output)

            if final_sentence is None:
                # 형식 깨짐
                return {
                    "uuid": script_uuid,
                    "input": script_tn,
                    "scriptITN": script_itn,
                    "output": cot_output,
                    "status": "ERROR_NO_FINAL_SENTENCE"
                }

            if final_sentence != script_itn.strip():
                # 정답 ITN과 불일치
                return {
                    "uuid": script_uuid,
                    "input": script_tn,
                    "scriptITN": script_itn,
                    "output": cot_output,
                    "status": "ERROR_FINAL_MISMATCH"
                }

            # 정상 케이스
            return {
                "uuid": script_uuid,
                "input": script_tn,
                "scriptITN": script_itn,
                "output": cot_output,
                "status": "OK"
            }

        except Exception as e:
            msg = str(e)
            # 400류 에러는 재시도해도 의미 없을 수 있으므로 메시지 보고 판단
            if "400" in msg or "BadRequest" in msg:
                return {
                    "uuid": script_uuid,
                    "input": script_tn,
                    "scriptITN": script_itn,
                    "output": f"ERROR_BAD_REQUEST: {msg}",
                    "status": "ERROR_BAD_REQUEST"
                }

            # 그 외에는 재시도 (429, 500 등)
            if i < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_SECONDS)
                continue

            # 재시도 모두 실패
            return {
                "uuid": script_uuid,
                "input": script_tn,
                "scriptITN": script_itn,
                "output": f"ERROR_MAX_RETRY: {msg}",
                "status": "ERROR_MAX_RETRY"
            }


# ============================
# 4. 메인 로직
# ============================

def main():
    print(f"입력 CSV 로드: {INPUT_CSV_PATH}")
    df = pd.read_csv(INPUT_CSV_PATH)

    required_cols = ["uuid", "scriptTN", "scriptITN"]
    for col in required_cols:
        if col not in df.columns:
            print(f"오류: '{col}' 컬럼이 존재하지 않습니다.")
            return

    tasks = []
    for _, row in df.iterrows():
        script_tn = str(row["scriptTN"])
        script_itn = str(row["scriptITN"])
        script_uuid = str(row["uuid"])
        tasks.append((script_tn, script_itn, script_uuid))

    print(f"총 샘플 수: {len(tasks)}")

    ok_count = 0
    total_count = 0

    # 파일을 미리 열어두고, 완료되는 대로 한 줄씩 기록
    # 재시작 시 이전 결과를 살리고 싶으면 "w" -> "a" 로 변경
    with open(OUTPUT_JSONL_PATH, "w", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(generate_cot_label, tn, itn, uuid): (tn, uuid)
                for tn, itn, uuid in tasks
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="CoT 생성 중"):
                try:
                    result = future.result()
                except Exception as e:
                    tn, uuid = futures[future]
                    result = {
                        "uuid": uuid,
                        "input": tn,
                        "scriptITN": None,
                        "output": f"ERROR_FUTURE_EXCEPTION: {e}",
                        "status": "ERROR_FUTURE_EXCEPTION"
                    }

                # 한 샘플 처리 결과를 바로 파일에 기록
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()

                total_count += 1
                if result.get("status") == "OK":
                    ok_count += 1

    print(f"JSONL 저장 완료: {OUTPUT_JSONL_PATH}")
    print(f"총 {total_count}개 중 OK: {ok_count}, ERROR: {total_count - ok_count}")

if __name__ == "__main__":
    main()