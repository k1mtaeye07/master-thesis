import os
import torch
import re
import json
import numpy as np

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from tqdm import tqdm

# ============================
# 1. 경로 및 하이퍼파라미터 설정
# ============================

# 1.1. 모델 경로 (기존과 동일)
BASE_MODEL_NAME = "/root/workspace/thesis-project/model/qwen3_model/Qwen3-1.7B"

# 1.2. 데이터셋 경로 (JSONL 파일로 변경)
# 학습/검증 데이터가 분리되어 있다면 각각 지정하고, 하나라면 split하여 사용합니다.
# 여기서는 생성된 cot_11900_dataset.jsonl을 사용한다고 가정합니다.
DATASET_PATH = "/workspace/data/cot_11900_dataset.jsonl"

# 1.3. 결과 저장 경로 (CoT 실험용 경로 구분)
OUTPUT_DIR = "/workspace/model/qwen3_1.7B_cot_finetuned"

# 1.4. 최대 시퀀스 길이 (CoT 길이를 고려하여 확장 권장)
# 분석 및 변환 과정이 포함되므로 1408 -> 2048로 상향 조정
MAX_SEQ_LENGTH = 2048 

# ============================
# 2. 프롬프트 템플릿 (CoT용)
# ============================

# Teacher 모델이 데이터를 생성할 때 사용한 시스템 프롬프트와 동일하게 유지하는 것이 핵심입니다.
# 그래야 모델이 '왜' 이런 형식으로 답해야 하는지 맥락을 이해하기 쉽습니다.
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

def create_chatml_prompt(example):
    """
    JSONL의 한 라인을 받아 Qwen ChatML 포맷으로 변환합니다.
    """
    # 1. 입력 (scriptTN)
    user_input = example['input']
    
    # 2. 출력 (CoT 전체 내용: 분석 -> 변환 -> 최종 문장)
    # Teacher 모델이 생성한 output 필드를 그대로 정답으로 사용합니다.
    model_output = example['output']

    # 3. ChatML 포맷팅
    prompt_text = (
        f"<|im_start|>system\n{COT_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n{model_output}<|im_end|>"
    )

    return {"text": prompt_text}

# ============================
# 3. 데이터셋 로딩 및 전처리
# ============================

print(f"데이터셋 로드 중: {DATASET_PATH}")
# JSONL 로드
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

# 3.1. 오류 데이터 필터링 (status가 OK인 것만 사용)
print(f"필터링 전 샘플 수: {len(dataset)}")
dataset = dataset.filter(lambda x: x['status'] == 'OK')
print(f"필터링 후 샘플 수 (Status=OK): {len(dataset)}")

# 3.2. Train/Validation Split (예: 95:5)
# 별도의 검증셋 파일이 없다면 여기서 나눕니다.
dataset = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = dataset['train']
eval_dataset = dataset['test']

print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

# 3.3. ChatML 포맷팅 적용
print("ChatML 포맷팅 적용 중...")
train_dataset = train_dataset.map(create_chatml_prompt, desc="Formatting Train")
eval_dataset = eval_dataset.map(create_chatml_prompt, desc="Formatting Eval")

print("샘플 전처리 결과(1건):", train_dataset[0])
print("샘플 전처리 결과(1건):", eval_dataset[0])

# ============================
# 4. 모델, 토크나이저, QLoRA 설정
# ============================

# 4.1. 4-bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 4.2. 모델 로드
print("모델 로딩 중...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2", 
)

# 4.3. 토크나이저 설정
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model.config.pad_token_id = tokenizer.eos_token_id

# 4.4. LoRA 설정 (기존 설정 유지)
peft_config = LoraConfig(
    r=32,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# ============================
# 5. Trainer 설정 및 학습
# ============================

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    run_name="qwen3_1.7B_cot_itn",
    push_to_hub=False,
    report_to="tensorboard",
    
    # 하이퍼파라미터 (기존과 유사하게 유지하되 배치 사이즈 등은 메모리에 맞춰 조정)
    num_train_epochs=3,
    per_device_train_batch_size=2,     # 메모리 부족 시 1로 조정
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=32,    # 배치 2 * 32 = 64 효과
    gradient_checkpointing=True,
    
    optim="paged_adamw_8bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_grad_norm=0.3,
    
    fp16=False,
    bf16=True,
    
    # 로깅 및 저장
    eval_strategy="steps",
    eval_steps=75,
    save_strategy="steps",
    save_steps=75,
    save_total_limit=3,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    # 시퀀스 설정
    max_length=MAX_SEQ_LENGTH,
    packing=True,  # 학습 효율을 위해 Packing 사용
    eos_token=tokenizer.eos_token,
    pad_token=tokenizer.pad_token,
    
    dataset_text_field="text",
    group_by_length=True,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    args=sft_config,
)

print("SFT Trainer 준비 완료. & 학습 시작...")

CHECKPOINT_PATH = "/workspace/model/qwen3_1.7B_cot_finetuned/checkpoint-156"
trainer.train(resume_from_checkpoint=CHECKPOINT_PATH)

# 학습 종료 후 저장
print("모델 저장 중...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"CoT 파인튜닝 완료: {OUTPUT_DIR}")