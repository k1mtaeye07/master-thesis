# Korean ASR Inverse Text Normalization using QLoRA Fine-Tuning and Prompt Engineering for a Small Language Model

> **Master's Thesis** | Soongsil University, Department of AI Techno Convergence | December 2025
> **Author:** Kim Tae-yeun (김태연) | **Advisor:** Prof. Jin-Hong Jeong (정진홍)

---

## 한국어 | Korean

> **QLoRA 파인튜닝 및 프롬프트 설계를 통한 경량언어모델의 한국어 음성인식 역정규화 기법**
> 이 연구는 단일 소비자용 GPU(RTX 4090) 환경에서 1.7B 규모의 경량언어모델(Qwen3-1.7B)과 QLoRA 파인튜닝, 구조적 Few-Shot 프롬프트 전략을 결합하여 한국어 ASR 역정규화(ITN) 시스템을 구축하고, 기존 RNN-Seq2Seq 베이스라인 대비 Overall-CER 기준 약 47% 성능 향상(3.34% → 1.77%)을 달성한 연구입니다.

---

## Table of Contents

- [Overview](#overview)
- [Key Contributions](#key-contributions)
- [System Architecture & Pipeline](#system-architecture--pipeline)
- [Experimental Setup](#experimental-setup)
- [Dataset](#dataset)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Notebooks](#notebooks)
- [Limitations & Future Work](#limitations--future-work)
- [Citation](#citation)

---

## Overview

**Inverse Text Normalization (ITN)** is a post-processing step for Automatic Speech Recognition (ASR) that converts spoken-form text into written-form text.

| Spoken Form (ASR Output) | Written Form (ITN Output) |
|---|---|
| 칠억 구천 오백 육십 일만 달러 | 7억 9,561만 달러 |
| 일억 원으로 딜리버리 사업을 이끌었는데, 일 년 새 매출이 서른네배나 늘었다 | 1억 원으로 딜리버리 사업을 이끌었는데, 1년 새 매출이 34배나 늘었다 |
| 오전 열한시쯤 약초를 캐려다 십 미터 아래로 추락해 열 아홉 시간 만에 구조됐다 | 오전 11시쯤 약초를 캐려다 10m 아래로 추락해 19시간 만에 구조됐다 |

**Why Korean ITN is challenging:**
- Korean is an agglutinative language — postpositions and endings attach directly to numerals, creating highly variable surface forms
- The same surface form can carry different meanings depending on context (e.g., "이" can mean the numeral 2, a demonstrative pronoun, or a subject marker)
- ASR outputs frequently contain spacing errors that compound the difficulty

**Why lightweight models matter:**
Large Language Models (LLMs, 70B+ params) achieve strong ITN performance but require expensive GPU clusters, making real-world deployment impractical. This work demonstrates that a **1.7B parameter SLM** with carefully designed training and prompting strategies can close that performance gap on a single consumer GPU.

---

## Key Contributions

1. **QLoRA-based lightweight training pipeline**
   Applied 4-bit NF4 quantization combined with Low-Rank Adaptation (LoRA rank=32) to reduce peak VRAM from 4.30 GB to **2.15 GB (50% reduction)**, enabling training on a single RTX 4090.

2. **CoT knowledge distillation with self-consistency filtering**
   Built a high-quality Chain-of-Thought (CoT) dataset by generating 3-step reasoning labels (Analyze → Transform → Output) from a large teacher model on a stratified subset of 11,900 samples. Applied self-consistency-based quality filtering to remove ~15% noisy examples.

3. **Structured Few-Shot prompt engineering**
   Designed a Markdown-based hierarchical prompt with negative constraints targeting the 7 most error-prone Korean numeric/pattern categories. This input-structure control approach outperforms the CoT approach in the SLM setting, confirming that **input structure control > internalized reasoning** for small models.

4. **Empirical evidence for SLM-optimal prompt strategy**
   Demonstrates that structured Few-Shot prompting is more effective than Few-Shot-CoT for 1.7B-scale models in format-sensitive conversion tasks — complementing prior work that focused on larger models.

---

## System Architecture & Pipeline

```
                        ┌─────────────────────────────────────────────────────┐
                        │              Proposed System Pipeline                │
                        └─────────────────────────────────────────────────────┘

  ASR Raw Transcript
  (Spoken-form text)
         │
         ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                         Data Construction                                       │
│                                                                                  │
│  AI-Hub Dataset (149,898 samples, 84 categories)                                │
│         │                                                                        │
│         ├──── Hierarchical Stratified Sampling (8:1:1 split)                   │
│         │     └── Balanced 11,900-sample distilled subset                       │
│         │                                                                        │
│         └──── Full Train Set (SFT with Few-Shot prompts)                        │
└────────────────────────────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
┌──────────────────────┐    ┌───────────────────────────────────┐
│  Path A: Few-Shot    │    │  Path B: Few-Shot-CoT             │
│  Guided SFT          │    │  (Teacher LLM-driven KD)          │
│                      │    │                                   │
│  Markdown-structured │    │  Teacher LLM                      │
│  input/output        │    │  ├── 3-step CoT label generation  │
│  examples injected   │    │  │   (Analyze → Transform → Out) │
│  into prompt         │    │  └── Self-consistency filtering   │
│                      │    │       (removes ~15% noisy data)   │
│  Max Seq: 1,408      │    │  Max Seq: 2,048                   │
│  Epochs: 1           │    │  Epochs: 3                        │
└──────────┬───────────┘    └──────────────┬────────────────────┘
           │                               │
           └───────────────┬───────────────┘
                           ▼
          ┌────────────────────────────────────┐
          │     QLoRA Adapter Framework        │
          │                                    │
          │  Base: Qwen3-1.7B (frozen, 4-bit)  │
          │  Adapter: LoRA (rank=32, α=16)     │
          │  Optimizer: Paged AdamW 32-bit     │
          │  Batch: 64 (gradient accumulation) │
          └────────────────┬───────────────────┘
                           ▼
          ┌────────────────────────────────────┐
          │       ITN Output                   │
          │  (Written-form, normalized text)   │
          └────────────────────────────────────┘
```

**Structured Few-Shot Prompt Design (7 core pattern categories):**

| # | Category | Example (Spoken → Written) |
|---|---|---|
| 1 | Month range | 일월부터 십이월 → 1월~12월 |
| 2 | Consecutive numbers | 일이삼사오 → 12345 |
| 3 | Large amounts | 칠억 구천만 원 → 7억 9,000만 원 |
| 4 | Mixed alphanumeric | 에이플러스 → A+ |
| 5 | Phone numbers | 공일공 일이삼사 오육칠팔 → 010-1234-5678 |
| 6 | Date/time expressions | 이천이십오년 십이월 → 2025년 12월 |
| 7 | Ordinal/idiomatic exceptions | 첫째, 둘째 → 첫째, 둘째 (no change) |

---

## Experimental Setup

### Hardware & Software

| Category | Item | Specification |
|---|---|---|
| Hardware | GPU | NVIDIA GeForce RTX 4090 (24GB VRAM) × 1 |
| Software | OS | Ubuntu 24.04 |
| | Framework | PyTorch 2.8.0, CUDA 12.8.1 |
| | Core Libraries | Transformers 4.57.1, PEFT 0.17.1, trl 0.25.0 |
| | Quantization | bitsandbytes 0.43.2 (4-bit NF4) |

### Hyperparameters

| Parameter | Value |
|---|---|
| Base Model | Qwen3-1.7B |
| Quantization | 4-bit (NF4) |
| LoRA Rank (r) / Alpha (α) | 32 / 16 |
| Learning Rate | 2×10⁻⁴ (Cosine Schedule) |
| Effective Batch Size | 64 (gradient accumulation) |
| Max Sequence Length | 2,048 (Few-Shot-CoT) / 1,408 (Few-Shot) |
| Optimizer | Paged AdamW (32-bit) |
| Epochs | 3 (Few-Shot-CoT) / 1 (Few-Shot) |

---

## Dataset

This project uses the **"숫자가 포함된 패턴 발화" (Numbers-included Pattern Utterances)** dataset from [AI-Hub](https://www.aihub.or.kr), provided by the National Information Society Agency (NIA).

> **Note:** The raw dataset must be downloaded directly from AI-Hub due to licensing restrictions. See [`data/README.md`](data/README.md) for download instructions.

| Split | Samples | Description |
|---|---|---|
| Train | ~119,918 | Full training set (SFT) |
| Validation | ~14,990 | Hyperparameter tuning |
| Test | ~14,990 | Final evaluation |
| Distilled subset | 11,900 | Stratified sample for CoT labeling (≈10% of train) |

**Dataset characteristics:**
- 84 sub-categories (cardinal numbers, dates, times, phone numbers, addresses, financial amounts, units, etc.)
- Long-tail distribution — hierarchical stratified sampling used to ensure balanced coverage
- 99% of samples are under 62 tokens (Qwen3 tokenizer), confirming no truncation occurs within the configured max sequence lengths

---

## Results

### Model Performance Comparison

| Model | Overall-CER (↓) | Numeric-CER (↓) | Target-CER (↓) | Span-F1 Macro (↑) | Span-F1 Micro (↑) |
|---|:---:|:---:|:---:|:---:|:---:|
| RNN-Seq2Seq (Baseline) | 3.34% | 0.19% | 42.0% | 0.19 | 0.20 |
| Qwen3-0.6B (no fine-tuning) | 72.48% | 1.05% | — | 0.01 | 0.01 |
| Qwen3-1.7B (no fine-tuning) | 53.08% | 0.73% | — | 0.04 | 0.05 |
| **Qwen3-1.7B + Few-Shot (Proposed)** | **1.77%** | 0.20% | **39.85%** | **0.21** | **0.22** |
| Qwen3-1.7B + Few-Shot-CoT | 4.60% | 0.25% | 45.0% | 0.18 | 0.19 |

**Metric definitions:**
- **Overall-CER**: Character Error Rate over the full sentence (Levenshtein distance-based)
- **Numeric-CER**: CER computed only on Arabic numeral characters (0–9)
- **Target-CER**: CER computed only on the target conversion spans (numbers, units, dates, etc.)
- **Span-F1**: Dice-coefficient F1 over predicted vs. ground-truth numeric spans; Macro = per-sample average, Micro = global aggregate

**Key findings:**
- The proposed Few-Shot model achieves **~47% reduction in Overall-CER** vs. the RNN baseline (3.34% → 1.77%)
- Structured Few-Shot outperforms Few-Shot-CoT on **all metrics** in the 1.7B model setting, supporting the hypothesis that input structure control is more effective than internalizing reasoning for small models
- Numeric-CER is comparable to the RNN baseline (0.20% vs 0.19%), with marginal residual hallucination as the only gap

### Resource Efficiency

| Model | Precision | Peak VRAM | Memory Saving | Training Speed |
|---|:---:|:---:|:---:|:---:|
| Standard LoRA (Baseline) | BF16 (16-bit) | 4.30 GB | — | 9.83 samples/s |
| **QLoRA (Proposed)** | NF4 (4-bit) | **2.15 GB** | **50.0%** | 6.75 samples/s |

> QLoRA reduces peak VRAM by 50% at the cost of ~31% slower training throughput — a favorable trade-off that enables larger batch sizes or longer sequences on constrained hardware.

---

## Repository Structure

```
korean-itn-qlora/
│
├── README.md
│
├── data/
│   ├── data_index.csv                       # Full dataset index (AI-Hub)
│   ├── train_set.csv                        # Training split
│   ├── validation_set.csv                   # Validation split
│   ├── test_set.csv                         # Test split
│   ├── golden_11900_stratified_samples.csv  # Stratified distilled subset (CoT input)
│   ├── cot_11900_dataset.jsonl              # Raw CoT-labeled data (teacher output)
│   └── cot_11900_dataset_filtered.jsonl     # Filtered CoT data (self-consistency passed)
│
├── src/
│   ├── CoT_label_generation.py              # Teacher model CoT label generation
│   ├── cot_11900_filtering.py               # Self-consistency quality filtering
│   ├── qrola_fine_tunning_1113.py           # Few-Shot QLoRA fine-tuning
│   ├── itn_vanilla_17_fine_tunned_CoT.py    # Few-Shot-CoT QLoRA fine-tuning
│   ├── itn_cot_17_tuned_evaluation.py       # Evaluation (CER / Span-F1)
│   └── seq2seq.py                           # RNN-Seq2Seq baseline
│
├── models/
│   └── qwen3_1.7B_1113/                     # Fine-tuned LoRA adapter weights
│
└── notebooks/
    ├── 01_data_analysis.ipynb               # Dataset distribution & token length analysis
    ├── 02_qlora_finetuning.ipynb            # QLoRA fine-tuning execution & training log
    ├── 03_cot_distillation.ipynb            # CoT label generation & filtering results
    └── 04_evaluation.ipynb                  # Performance comparison & result tables
```

---

## Notebooks

Execution results are documented in the notebooks below. Each notebook contains inline outputs so you can review results without re-running.

| Notebook | Description |
|---|---|
| [`01_data_analysis.ipynb`](notebooks/01_data_analysis.ipynb) | Dataset category distribution (84 patterns), token length box plot, train/val/test split statistics |
| [`02_qlora_finetuning.ipynb`](notebooks/02_qlora_finetuning.ipynb) | QLoRA fine-tuning run on Qwen3-1.7B — training loss curve, VRAM usage log (2.15 GB peak) |
| [`03_cot_distillation.ipynb`](notebooks/03_cot_distillation.ipynb) | Teacher model CoT label generation on the 11,900-sample distilled subset, self-consistency filtering results (~15% removed) |
| [`04_evaluation.ipynb`](notebooks/04_evaluation.ipynb) | Full model comparison across all 5 metrics, Span-F1 bar chart, sample predictions vs. ground truth |

---

## Limitations & Future Work

1. **Residual numeric hallucination:** The generative model occasionally drops or alters digits, resulting in slightly worse Numeric-CER than the deterministic RNN baseline (0.20% vs 0.19%). A hybrid post-processing module (e.g., rule-based numeric verification) is planned.

2. **Inference latency:** Autoregressive generation is slower than WFST/RNN for real-time use. Future work includes conversion to ONNX / TensorRT and application of speculative decoding.

3. **Robustness to ASR errors:** The model was trained on clean AI-Hub text data. Noise injection training with simulated ASR error patterns is needed for production deployment.

---

## Citation

If you use this code or find this work helpful, please cite:

```bibtex
@mastersthesis{kim2025korean,
  title     = {Korean ASR Inverse Text Normalization using QLoRA Fine-Tuning and Prompt Engineering for a Small Language Model},
  author    = {Kim, Tae-yeun},
  school    = {Soongsil University},
  year      = {2025},
  month     = {December},
  address   = {Seoul, Republic of Korea},
  note      = {Department of AI Techno Convergence}
}
```

---

## Acknowledgements

This research was supported by the guidance of **Prof. Jin-Hong Jeong** (advisor) and the insightful feedback of **Prof. Gye-young Kim** and **Prof. Kwang-young Park** (thesis committee). The dataset was provided by the National Information Society Agency (NIA) via [AI-Hub](https://www.aihub.or.kr).

---

<p align="center">
  <sub>Master's Thesis · Soongsil University · Department of AI Techno Convergence · 2025</sub>
</p>
