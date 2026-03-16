import pandas as pd
import re
import jiwer
import ast
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import numpy as np

# JIWER의 경고 메시지(빈 문자열 비교)를 숨깁니다.
warnings.filterwarnings("ignore", category=UserWarning, module='jiwer')

# --- 1. 설정 및 상수 ---
TEST_FILE_PATH = "/workspace/data/test_set_preprocessed.csv"
VLLM_API_URL = "http://localhost:10000/v1"
MODEL_NAME = "qwen3-1.7B-tuned_cot" # vLLM 서버에 배포된 최종 CoT 모델 이름

client = OpenAI(api_key="vllm", base_url=VLLM_API_URL)

# CoT SYSTEM PROMPT: 학습에 사용했던 상세 지침을 그대로 사용하여 CoT 출력을 유도합니다.
COT_SYSTEM_PROMPT = """
당신은 한국어 **텍스트 역정규화(Inverse Text Normalization, ITN)**를 전문으로 하는 AI assistant입니다.
당신의 임무는 **[입력 문장]**을 **[한국어 숫자 발화 및 패턴 지침]**에 따라 분석하고,
그 과정을 **[출력 형식]**에 맞춰 정확하게 서술한 뒤,
지침에 부합하는 **최종 변환 문장**을 생성하는 것입니다.

아래의 규칙과 출력 형식을 반드시 준수하십시오.
임의로 내용을 추가하거나 가정하거나 추론하지 마십시오.
[입력 문장]에 존재하지 않는 정보는 절대로 만들어내지 마십시오.

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
   - [입력 문장]에서 숫자/패턴/단위 표현을 모두 식별.
   - 각 표현이 어떤 한국어 발화 특성을 가지는지 설명.
   - 예외 표현이 있다면 그대로 유지해야 하는 근거 설명.
   - (주의) 입력에 없는 정보는 만들지 말 것.

### 2. 변환:
   - 식별된 각 표현을 지침의 몇 번 규칙에 따라 어떻게 변환하는지 단계적으로 설명.
   - 예외 표현은 “변환하지 않는다”고 이유와 함께 명시.
   - hallucination 금지: 존재하지 않는 표현, 숫자, 단어 추가 금지.

### 3. 최종 문장:
   - 변환이 완료된 문장을 중괄호 { } 안에 한 번만 제시.
   - 절대로 새로운 변환 결과를 생성하지 말 것.
   - 예: {12만 1,169명 중 9만 3,939명이 남성이었다.}
"""


# --- 2. 평가 지표 계산 헬퍼 함수 ---

def get_digits(text: str) -> str:
    """텍스트에서 숫자(0-9)만 추출하여 문자열로 반환"""
    if not text:
        return ""
    return "".join(re.findall(r'[0-9]', str(text)))

def get_numeric_spans_from_text(text: str) -> set:
    """[예측용] 텍스트에서 '숫자가 포함된' 모든 어절(span)을 추출하여 set으로 반환"""
    if not text:
        return set()
    spans = re.findall(r'\S*[\d]\S*', str(text))
    return set(spans)

def safe_cer(truth: str, pred: str) -> float:
    """CER 계산 시 빈 문자열 예외 처리"""
    if not truth and not pred:
        return 0.0
    if not truth or not pred:
        return 1.0
    return jiwer.cer(truth, pred)

def parse_script_number_word(span_str: str) -> list:
    """'scriptNumberWord' 컬럼의 문자열을 파이썬 리스트로 안전하게 파싱"""
    if not span_str or span_str == "[]":
        return []
    try:
        return ast.literal_eval(span_str)
    except (ValueError, SyntaxError):
        print(f"경고: scriptNumberWord 파싱 오류. '{span_str}'")
        return []

def calculate_single_row_metrics(
    ground_truth_itn: str, 
    prediction: str, 
    ground_truth_span_str: str
) -> dict:
    """단일 row에 대해 'scriptNumberWord' 기준으로 모든 지표를 계산합니다."""
    
    overall_cer = safe_cer(ground_truth_itn, prediction)
    
    truth_spans_list = parse_script_number_word(ground_truth_span_str)
    truth_spans_set = set(truth_spans_list)
    truth_span_text = " ".join(sorted(truth_spans_list)) 
    truth_digit_str = " ".join(get_digits(truth_span_text)) 
    
    pred_spans_set = get_numeric_spans_from_text(prediction)
    pred_span_text = " ".join(sorted(list(pred_spans_set))) 
    pred_digit_str = " ".join(get_digits(pred_span_text)) 

    numeric_cer = safe_cer(truth_digit_str, pred_digit_str)
    target_cer = safe_cer(truth_span_text, pred_span_text)

    # Span F1 (TP, FP, FN 계산)
    tp = len(truth_spans_set & pred_spans_set)
    fp = len(pred_spans_set - truth_spans_set)
    fn = len(truth_spans_set - truth_spans_set)
    
    fn = len(truth_spans_set - pred_spans_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    span_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "Overall-CER": overall_cer,
        "Numeric-CER": numeric_cer,
        "Target-CER": target_cer,
        "Span-F1": span_f1,
        "Span-Precision": precision,
        "Span-Recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


# --- 3. CoT 결과에서 최종 ITN 문장 추출 ---
def extract_final_sentence(cot_output: str) -> str:
    # 1. '3. 최종 문장:' 패턴을 찾아 그 뒤의 모든 내용 (.*)을 추출
    # DOTALL 플래그를 사용하여 출력이 여러 줄이어도 모두 처리
    m = re.search(r"3\.\s*최종 문장\s*:\s*(.*)", cot_output, re.DOTALL)
    
    if m:
        # 추출된 원시 텍스트 (앞뒤 공백 포함 가능)
        extracted = m.group(1).strip()
        
        # 2. [핵심 수정] 추출된 내용에서 앞뒤의 중괄호 {}만 제거하고, 일반적인 공백 정리
        #    - strip('{}'): 앞뒤 중괄호만 제거 (문장 중간의 중괄호는 남김)
        #    - strip(): 일반적인 앞뒤 공백 제거
        return extracted.strip('{}').strip()
    
    # 패턴을 찾지 못한 경우 (fallback)
    return cot_output.splitlines()[0] if cot_output else ""


# --- 4. vLLM 추론 함수 (raw_output 반환 추가) ---
def get_vllm_prediction(input_text: str) -> tuple[str, str]:
    """vLLM 서버에 요청 후, 원시 CoT 출력과 최종 ITN 결과 두 가지를 반환."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": COT_SYSTEM_PROMPT},
                {"role": "user", "content": input_text}
            ],
            temperature=0,
            max_tokens=2048,
        )
        raw_output = completion.choices[0].message.content.strip()
        final_itn_result = extract_final_sentence(raw_output)
        
        # 원시 출력과 추출된 결과를 함께 반환
        return raw_output, final_itn_result

    except Exception as e:
        print(f"API Error: {e}")
        return "", ""

# --- 5. 메인 실행 로직 ---
def main():
    print(f"'{TEST_FILE_PATH}'에서 테스트 데이터 로드 중...")
    try:
        df = pd.read_csv(TEST_FILE_PATH)
    except FileNotFoundError:
        print(f"오류: 테스트 파일을 찾을 수 없습니다. 경로를 확인하세요: {TEST_FILE_PATH}")
        return
    
    required_cols = ['uuid', 'scriptTN', 'scriptITN', 'scriptNumberWord']
    if not all(col in df.columns for col in required_cols):
        print(f"오류: {required_cols} 컬럼 중 일부가 없습니다.")
        print(f"현재 컬럼: {df.columns.tolist()}")
        return

    df = df.dropna(subset=['scriptTN', 'scriptITN'])
    df = df[df['scriptTN'].str.strip() != '']
    df['scriptNumberWord'] = df['scriptNumberWord'].fillna('[]')

    inputs = df['scriptTN'].tolist()
    
    # 원시 CoT 출력과 최종 예측값 저장을 위한 리스트 준비
    raw_cot_outputs = [None] * len(inputs)
    predictions = [None] * len(inputs)

    print(f"총 {len(inputs)}건의 테스트 데이터에 대해 vLLM({MODEL_NAME}) CoT 추론 시작...")
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_index = {executor.submit(get_vllm_prediction, text): i for i, text in enumerate(inputs)}
        
        for future in tqdm(as_completed(future_to_index), total=len(inputs), desc="추론 진행 중"):
            index = future_to_index[future]
            # 반환된 튜플을 각각 저장
            raw_output, final_prediction = future.result() 
            raw_cot_outputs[index] = raw_output
            predictions[index] = final_prediction

    # 원시 CoT 출력과 최종 예측값 컬럼 추가
    df['raw_cot_output'] = raw_cot_outputs
    df['prediction'] = predictions
    print("추론 완료. Row별 성능 지표 계산 중...")

    # Row별 지표 계산
    results = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Row별 지표 계산 중"):
        metrics = calculate_single_row_metrics(
            row['scriptITN'], 
            row['prediction'], 
            row['scriptNumberWord'] 
        )
        results.append(metrics)

    metrics_df = pd.DataFrame(results)
    final_df = pd.concat([df, metrics_df], axis=1)

    # --- CSV 저장 ---
    output_columns = [
        'uuid', 
        'scriptITN',  
        'scriptNumberWord', 
        'scriptTN',   
        'raw_cot_output',   # <-- CoT 전체 출력
        'prediction',       # 추출된 ITN 결과
        'Overall-CER', 
        'Numeric-CER', 
        'Target-CER', 
        'Span-F1', 
        'Span-Precision', 
        'Span-Recall'
    ]
    
    output_filename = f"model_results_{MODEL_NAME.replace('/', '_')}_cot_1120.csv"
    
    final_df[output_columns].to_csv(output_filename, index=False, encoding="utf-8-sig")
    print(f"Row별 상세 결과가 '{output_filename}' 파일로 저장되었습니다.")

    # --- 전체 요약 통계 ---
    print(f"\n--- [{MODEL_NAME} 전체 요약 통계] ---")
    
    print("📊 [Macro-Averages (Row별 지표의 평균)]")
    macro_avg_metrics = final_df[['Overall-CER', 'Numeric-CER', 'Target-CER', 'Span-F1', 'Span-Precision', 'Span-Recall']].mean()
    print(macro_avg_metrics)
    
    # Micro-Averages (전체 Span 합산 기준) 계산
    total_tp = final_df['tp'].sum()
    total_fp = final_df['fp'].sum()
    total_fn = final_df['fn'].sum()

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    print("\n" + "---" + "\n")
    print("📈 [Micro-Averages (전체 Span 합산 기준 - F1 표준)]")
    print(f"  Span-F1 (Micro Avg): {micro_f1:.4f}")
    print(f"  (Span-Precision: {micro_precision:.4f})")
    print(f"  (Span-Recall: {micro_recall:.4f})")
    print("-------------------------------------------------")

    """
    11/19
    --- [qwen3-1.7B-tuned_cot 전체 요약 통계] ---
    📊 [Macro-Averages (Row별 지표의 평균)]
    Overall-CER       0.146927
    Numeric-CER       0.271306
    Target-CER        0.493314
    Span-F1           0.175747
    Span-Precision    0.178800
    Span-Recall       0.176908
    dtype: float64

    ---

    📈 [Micro-Averages (전체 Span 합산 기준 - F1 표준)]
    Span-F1 (Micro Avg): 0.1869
    (Span-Precision: 0.1781)
    (Span-Recall: 0.1966)
    -------------------------------------------------

    11/20
    --- [qwen3-1.7B-tuned_cot 전체 요약 통계] ---
    📊 [Macro-Averages (Row별 지표의 평균)]
    Overall-CER       0.045968
    Numeric-CER       0.251218
    Target-CER        0.455212
    Span-F1           0.183108
    Span-Precision    0.186390
    Span-Recall       0.184244
    dtype: float64

    ---

    📈 [Micro-Averages (전체 Span 합산 기준 - F1 표준)]
    Span-F1 (Micro Avg): 0.1980
    (Span-Precision: 0.1912)
    (Span-Recall: 0.2053)
    -------------------------------------------------
    """
    
if __name__ == "__main__":
    main()