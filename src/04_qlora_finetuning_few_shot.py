import os
import torch
import re
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from tqdm import tqdm

# --- 1. (필수) 모델 및 경로 설정 ---
# ------------------------------------

# 1.1. 원본 모델이 저장된 로컬 경로 (Git LFS로 클론한 경로)
BASE_MODEL_NAME = "/root/workspace/thesis-project/model/qwen3_model/Qwen3-1.7B"

# 1.2. 훈련/검증 데이터셋 경로
TRAIN_CSV_PATH = "/root/workspace/thesis-project/data/train_set.csv"
VALID_CSV_PATH = "/root/workspace/thesis-project/data/validation_set.csv"

# 1.3. [중요] 훈련 결과물(체크포인트)이 저장될 경로
OUTPUT_DIR = "/workspace/model/qwen3_1.7B_1113"

# 1.4. [중요] 최대 시퀀스 길이
# (예시) Max: 1349 → 64의 배수로 1408 설정
MAX_SEQ_LENGTH = 1408
# ------------------------------------

# 1.5. 데이터 전처리 시 불필요한 대괄호 제거용 헬퍼 함수
def remove_brackets(text: str) -> str:
    # '[내용]' 패턴에서 대괄호만 제거. 내용은 남김.
    return re.sub(r"\[([^\]]+)\]", r"\1", text)

# --- 2. ChatML 프롬프트 템플릿 함수 ---
# ------------------------------------
def create_chatml_prompt(example):
    # 2.1. (중요) 모델의 역할을 규정하는 시스템 프롬프트
    system_prompt = (
        '''
        당신은 한국어 텍스트 **역정규화(Inverse Text Normalization, ITN)**를 전문으로 하는 AI assistant입니다. 
        주어진 **구어체 문장을 의미를 바꾸지 않고, [한국어 역정규화 지침 대상]이 아닌 텍스트는 입력 형태 그대로 출력**해야 합니다. 숫자 기반 문어체만 **정확하게 역정규화** 하세요.

        분석·해설·사고과정(CoT)·태그(<think> 등)을 출력하지 말고, 최종 변환 결과 한 줄만 출력하시오.
        변환 시, 다음의 [한국어 역정규화 지침]을 **반드시** 준수해야 합니다.

        ## [한국어 역정규화 지침]
        ### 1. 아라비아 숫자로 변환 (필수)
        * **날짜 및 시간:**
            * *예시:* "십 일 월 십 일에서 십 일 일까지 일 박 이 일로" -> "11월 10일에서 11일까지 1박 2일로"
            * *예시:* "일 에서 구 월만 놓고 봤을 때" -> "1~9월만 놓고 봤을 때"
            * *예시:* "네 시 이십 분이 돼서" -> "4시 20분이 돼서"
        * **정량적 숫자 (수량, 단위, 금액 등):**
            * **일반 수량/단위:** 표준 단위(e.g., `kg`, `cm`, `km/h`, `kcal`)를 사용합니다.
                * *예시:* "껍질을 깐 오렌지 한 개 칼로리는 대략 팔십 킬로칼로리 정도 되는데" -> "껍질을 깐 오렌지 1개 칼로리는 대략 80kcal 정도 되는데"
                * *예시:* "톰 브라운의 티비 구 공 삼 안경은 안구 사이즈 [사십 구 밀리미터]" -> "톰 브라운의 TB-903 안경은 안구 사이즈 49mm"
            * **금액 및 큰 숫자:** '만', '억', '조' 단위는 한글로 유지하되, 그 앞의 숫자는 아라비아 숫자로 표기하고 4자리마다 쉼표(,)를 추가합니다.
                * *예시:* "십 이만 천 백 육십 구 명 중 구만 삼천 구백 삼십 구 명이 남성이었다." -> "12만 1,169명 중 9만 3,939명이 남성이었다."
                * *예시:* "순이익은 구억 삼천 이백만 원으로 전년 동기 일억 삼천 삼백만 원 대비 육백 점 팔 퍼센트 증가했다." -> "순이익은 9억 3,200만 원으로 전년 동기 1억 3,300만 원 대비 600.8% 증가했다."
        * **혼합 패턴 (카드/전화/주민번호 등):**
            * *예시:* "공 하나 오 하나 사 오 둘 하나 하나 육 오 로 전화해서" -> "015-1452-1165로 전화해서"
            * *예시:* "주민번호가 구 이 공 공 공 팔 삼 삼 칠 육 이 팔 구인데" -> "주민번호가 920008-3376289인데"
        ### 2. 변환 예외 (중요)
        * **고유 기수:**
            * *예시:* "첫 번째" -> "첫 번째" (O)
            * *예시:* "스무 번째" -> "스무 번째" (O)
            * *예시:* "일흔여섯 살" -> "일흔여섯 살" (O)
            * *(오류):* "첫 번째" -> "1 번째" (X)
        * **관용구 및 모호한 표현:**
            * *예시:* "한두 번이 아니다" -> "한두 번이 아니다" (O)
            * *예시:* "대여섯 명" -> "대여섯 명" (O)
        ### 3. 공통 지침
        한국어와 영어 외의 언어는 답변으로 포함 금지합니다.
        ## [출력 형식]
        * 불필요한 설명, 사과, 서론을 모두 제외하고 **변환된 문어체 결과만** 출력합니다.
        '''
    )

    # 2.2. CSV의 'scriptTN'(입력), 'scriptITN'(정답) 컬럼을 사용
    user_input = remove_brackets(example['scriptTN'])  # 구어체 (Input)
    model_output = example['scriptITN']                # 문어체 (Target)

    # 2.3. Qwen3 ChatML
    prompt_text = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n{model_output}<|im_end|>"
    )

    # 2.4. SFTTrainer가 사용할 'text' 필드로 반환
    return {"text": prompt_text}

print("ChatML 프롬프트 템플릿 준비 완료.")

# --- 3. 데이터셋 로딩 및 전처리 ---
# ------------------------------------
data_files = {
    "train": TRAIN_CSV_PATH,
    "validation": VALID_CSV_PATH
}
raw_datasets = load_dataset("csv", data_files=data_files)

print("데이터 전처리 및 포맷팅 (ChatML → text 컬럼)…")
# ⚠️ 리스트로 덮어쓰지 말고, Dataset.map으로 Dataset → Dataset 유지
raw_datasets = raw_datasets.map(
    create_chatml_prompt,
    remove_columns=raw_datasets["train"].column_names,  # 원본 컬럼 제거하고 text만 남김(원하면 제거)
    desc="Formatting to ChatML `text`",
)

print("샘플 전처리 결과(1건):", raw_datasets["train"][0])

# --- 4. 모델 및 토크나이저, QLoRA(Peft) 설정 ---
# ------------------------------------

# 4.1. QLoRA를 위한 4-bit 양자화 설정 (BitsAndBytes)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 4.2. 기본 모델 로드
print("4-bit QLoRA 모델 로딩 중...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",     # 환경에 맞춰 auto/0 선택
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
print("모델 로딩 완료.")

# 4.3. 토크나이저 로드 및 PAD/EOS 정렬
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model.config.pad_token_id = tokenizer.eos_token_id
print("토크나이저 로딩 및 PAD/EOS 토큰 설정 완료.")

# 4.5. PEFT LoRA 설정
qwen_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
peft_config = LoraConfig(
    r=32,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=qwen_target_modules,
)
print("PEFT LoRA 설정 완료.")

# --- 5. SFT 설정 및 Trainer ---
# ------------------------------------
sft_config = SFTConfig(
    # 저장/런관리
    output_dir=OUTPUT_DIR,
    run_name="qwen3_1.7B_itn_qlora_1113",
    push_to_hub=False,
    report_to="tensorboard",

    # 학습 스케줄/옵티마이저
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=32,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_grad_norm=0.3,
    fp16=False,
    bf16=True,

    # 평가/저장/로그
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # 시퀀스/토큰
    max_length=MAX_SEQ_LENGTH,
    packing=True,
    eos_token=tokenizer.eos_token,
    pad_token=tokenizer.pad_token,

    # 본 리팩토링의 핵심: text 컬럼 지정
    dataset_text_field="text",
    group_by_length=True,

    # (선택) 어시스턴트 부분만 로스 계산
    # assistant_only_loss=True,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=raw_datasets["train"],
    eval_dataset=raw_datasets["validation"],
    peft_config=peft_config,
    args=sft_config,
)

print("SFT Trainer 준비 완료.")

# --- 6. 훈련 시작 ---
# ------------------------------------
print(f"최종 설정: MAX_SEQ_LENGTH={MAX_SEQ_LENGTH}, r=32, optim=paged_adamw_8bit")
print("훈련(Training)을 시작합니다...")

trainer.train()

# 6.1. 훈련 완료 후 최종 어댑터(Adapter) 저장
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("--- 훈련 완료 및 모델 저장 완료 ---")
