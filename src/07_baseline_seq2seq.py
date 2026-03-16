import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import pandas as pd
import numpy as np
import random
import time
import math
from tqdm import tqdm

import re
import jiwer
import ast
import warnings

# --- [!! 신규 !!] ---
# 텐서보드 로깅을 위한 SummaryWriter 임포트
from torch.utils.tensorboard import SummaryWriter
# --------------------

warnings.filterwarnings("ignore", category=UserWarning, module='jiwer')

# --- 1. Configuration ---
# (이전과 동일)
TRAIN_FILE_PATH = "/root/workspace/thesis-project/data/train_set_preprocessed.csv" 
VALID_FILE_PATH = "/root/workspace/thesis-project/data/validation_set_preprocessed.csv"
TEST_FILE_PATH = "/root/workspace/thesis-project/data/test_set_preprocessed.csv"  
EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
CLIP = 1
BATCH_SIZE = 64
N_EPOCHS = 10 
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

# --- 2. Vocabulary & Tokenizer ---
# (이전과 동일)
def tokenize_kr(text):
    return list(text)

class Vocabulary:
    def __init__(self):
        self.special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        self.stoi = {token: i for i, token in enumerate(self.special_tokens)}
        self.itos = {i: token for i, token in enumerate(self.special_tokens)}
        self.pad_idx = self.stoi[PAD_TOKEN]
        self.sos_idx = self.stoi[SOS_TOKEN]
        self.eos_idx = self.stoi[EOS_TOKEN]
        self.unk_idx = self.stoi[UNK_TOKEN]

    def build_vocab(self, texts):
        idx = len(self.special_tokens)
        char_counts = {}
        for text in texts:
            for char in tokenize_kr(text):
                char_counts[char] = char_counts.get(char, 0) + 1
        for char, count in char_counts.items():
            if count > 1 and char not in self.stoi:
                self.stoi[char] = idx
                self.itos[idx] = char
                idx += 1
    def numericalize(self, text):
        tokenized = tokenize_kr(text)
        return [self.sos_idx] + [self.stoi.get(char, self.unk_idx) for char in tokenized] + [self.eos_idx]
    def __len__(self):
        return len(self.itos)

# --- 3. Dataset & DataLoader ---
# (이전과 동일)
class ITNDataset(Dataset):
    def __init__(self, df, src_vocab, trg_vocab):
        self.src_texts = df['scriptTN'].tolist() 
        self.trg_texts = df['scriptITN'].tolist()
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
    def __len__(self):
        return len(self.src_texts)
    def __getitem__(self, idx):
        src_numerical = self.src_vocab.numericalize(self.src_texts[idx])
        trg_numerical = self.trg_vocab.numericalize(self.trg_texts[idx])
        return torch.LongTensor(src_numerical), torch.LongTensor(trg_numerical)

def create_collate_fn(pad_idx):
    def collate_fn(batch):
        src_batch, trg_batch = zip(*batch)
        src_lengths = [len(s) for s in src_batch]
        trg_lengths = [len(t) for t in trg_batch]
        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
        trg_padded = pad_sequence(trg_batch, batch_first=True, padding_value=pad_idx)
        return src_padded, torch.LongTensor(src_lengths), trg_padded, torch.LongTensor(trg_lengths)
    return collate_fn

# --- 4. Model Definition (Seq2Seq with Attention) ---
# (이전과 동일)
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src, src_len):
        embedded = self.dropout(self.embedding(src))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len.to('cpu'), batch_first=True, enforce_sorted=False)
        packed_outputs, hidden = self.rnn(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear((hid_dim * 2) + hid_dim, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)
    def forward(self, hidden, encoder_outputs, mask):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        attention = attention.masked_fill(mask == 0, -1e10)
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim + (hid_dim * 2), hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2 + hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, input, hidden, encoder_outputs, mask):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden, encoder_outputs, mask).unsqueeze(1)
        context = torch.bmm(a, encoder_outputs)
        rnn_input = torch.cat((embedded, context), dim=2)
        rnn_hidden = hidden.unsqueeze(0).repeat(self.n_layers, 1, 1)
        output, hidden = self.rnn(rnn_input, rnn_hidden)
        hidden = hidden[-1]
        embedded = embedded.squeeze(1)
        output = output.squeeze(1)
        context = context.squeeze(1)
        prediction = self.fc_out(torch.cat((output, context, embedded), dim=1))
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device
    def create_mask(self, src):
        mask = (src != self.src_pad_idx)
        return mask
    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src, src_len)
        mask = self.create_mask(src)
        input = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs, mask)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
        return outputs

# --- 5. Training & Evaluation ---
# (이전과 동일)
def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(tqdm(iterator, desc="Training")):
        src, src_len, trg, _ = batch
        src, trg = src.to(DEVICE), trg.to(DEVICE)
        optimizer.zero_grad()
        output = model(src, src_len, trg)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(iterator, desc="Evaluating")):
            src, src_len, trg, _ = batch
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            output = model(src, src_len, trg, 0) # No teacher forcing
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# --- 6. Inference & Metrics ---
# (이전과 동일, 학생의 LLM 평가 로직)
def translate_sentence(model, src_text, src_vocab, trg_vocab, device, max_len=150):
    model.eval()
    tokens = src_vocab.numericalize(src_text)
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    src_len = torch.LongTensor([len(tokens)]).to('cpu')
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)
    mask = model.create_mask(src_tensor)
    trg_indexes = [trg_vocab.sos_idx]
    for _ in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs, mask)
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == trg_vocab.eos_idx:
            break
    trg_tokens = [trg_vocab.itos[i] for i in trg_indexes if i not in [trg_vocab.pad_idx, trg_vocab.sos_idx, trg_vocab.eos_idx]]
    return "".join(trg_tokens)

def get_digits(text: str) -> str:
    if not text: return ""
    return "".join(re.findall(r'[0-9]', str(text)))
def get_numeric_spans_from_text(text: str) -> set:
    if not text: return set()
    spans = re.findall(r'\S*[\d]\S*', str(text))
    return set(spans)
def safe_cer(truth: str, pred: str) -> float:
    if not truth and not pred: return 0.0
    if not truth or not pred: return 1.0
    return jiwer.cer(truth, pred)
def parse_script_number_word(span_str: str) -> list:
    if not span_str or span_str == "[]": return []
    try: return ast.literal_eval(span_str)
    except (ValueError, SyntaxError): return []
def calculate_single_row_metrics(ground_truth_itn: str, prediction: str, ground_truth_span_str: str) -> dict:
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
    tp = len(truth_spans_set & pred_spans_set)
    fp = len(pred_spans_set - truth_spans_set)
    fn = len(truth_spans_set - pred_spans_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    span_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {"Overall-CER": overall_cer, "Numeric-CER": numeric_cer, "Target-CER": target_cer,
            "Span-F1": span_f1, "Span-Precision": precision, "Span-Recall": recall,
            "tp": tp, "fp": fp, "fn": fn}

# --- 7. Main Execution ---
# [!! 수정 !!] 텐서보드 로거(writer) 추가
# -------------------------
def main():
    print("--- 1. Loading Data ---")
    try:
        train_df = pd.read_csv(TRAIN_FILE_PATH)
        valid_df = pd.read_csv(VALID_FILE_PATH)
        test_df = pd.read_csv(TEST_FILE_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check ..._FILE_PATH variables.")
        return

    required_cols = ['uuid', 'scriptTN', 'scriptITN', 'scriptNumberWord']
    if not all(col in test_df.columns for col in required_cols):
        print(f"오류: {required_cols} 컬럼 중 일부가 테스트셋에 없습니다.")
        print(f"현재 컬럼: {test_df.columns.tolist()}")
        return
    train_df = train_df.dropna(subset=['scriptTN', 'scriptITN'])
    valid_df = valid_df.dropna(subset=['scriptTN', 'scriptITN'])
    test_df = test_df.dropna(subset=['scriptTN', 'scriptITN'])
    test_df['scriptNumberWord'] = test_df['scriptNumberWord'].fillna('[]')
    print(f"Loaded {len(train_df)} train, {len(valid_df)} valid, {len(test_df)} test samples.")

    print("--- 2. Building Vocabularies ---")
    src_vocab = Vocabulary()
    trg_vocab = Vocabulary()
    src_vocab.build_vocab(train_df['scriptTN'])
    trg_vocab.build_vocab(train_df['scriptITN'])
    INPUT_DIM = len(src_vocab)
    OUTPUT_DIM = len(trg_vocab)
    SRC_PAD_IDX = src_vocab.pad_idx
    TRG_PAD_IDX = trg_vocab.pad_idx
    print(f"Source Vocab Size: {INPUT_DIM}")
    print(f"Target Vocab Size: {OUTPUT_DIM}")

    print("--- 3. Creating DataLoaders ---")
    train_dataset = ITNDataset(train_df, src_vocab, trg_vocab)
    valid_dataset = ITNDataset(valid_df, src_vocab, trg_vocab)
    collate_fn = create_collate_fn(SRC_PAD_IDX) 
    train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_iterator = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print("--- 4. Initializing Model ---")
    attn = Attention(HID_DIM)
    enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, attn)
    model = Seq2Seq(enc, dec, SRC_PAD_IDX, DEVICE).to(DEVICE)
    model.apply(init_weights)
    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    # --- [!! 신규 !!] ---
    # 텐서보드 로거(writer) 초기화. 'runs/rnn_seq2seq_baseline' 폴더에 저장됩니다.
    writer = SummaryWriter('runs/rnn_seq2seq_baseline')
    # --------------------

    print("--- 5. Starting Training ---")
    best_valid_loss = float('inf')
    model_save_path = 'rnn-itn-baseline-model.pt'

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        
        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        # [콘솔 출력]
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

        # --- [!! 신규 !!] ---
        # 텐서보드에 스칼라 값(손실, PPL) 기록
        writer.add_scalar('Loss/train', train_loss, epoch + 1)
        writer.add_scalar('Perplexity/train', math.exp(train_loss), epoch + 1)
        writer.add_scalar('Loss/valid', valid_loss, epoch + 1)
        writer.add_scalar('Perplexity/valid', math.exp(valid_loss), epoch + 1)
        # --------------------
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_save_path)
            print(f'\t -> Best model saved to {model_save_path}!')

    writer.close() # --- [!! 신규 !!] --- 훈련 종료 후 writer 닫기
    
    print("--- 6. Starting Final Test ---")
    # (이하 평가 로직은 이전과 동일)
    model.load_state_dict(torch.load(model_save_path))
    
    results = [] 
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Testing (Row-by-Row)"):
        src_text = row['scriptTN']
        ref_text = row['scriptITN']
        span_str = row['scriptNumberWord']
        
        pred_text = translate_sentence(model, src_text, src_vocab, trg_vocab, DEVICE)
        
        metrics = calculate_single_row_metrics(ref_text, pred_text, span_str)
        
        metrics['uuid'] = row['uuid']
        metrics['scriptITN'] = ref_text
        metrics['scriptNumberWord'] = span_str
        metrics['scriptTN'] = src_text
        metrics['prediction'] = pred_text
        
        results.append(metrics)

    print("\n--- 7. Final Results (RNN/GRU Seq2Seq) ---")
    metrics_df = pd.DataFrame(results)
    output_columns = [
        'uuid', 'scriptITN', 'scriptNumberWord', 'scriptTN', 'prediction',
        'Overall-CER', 'Numeric-CER', 'Target-CER', 'Span-F1', 
        'Span-Precision', 'Span-Recall'
    ]
    output_filename = f"model_results_RNN_SEQ2SEQ_per_row.csv"
    metrics_df[output_columns].to_csv(output_filename, index=False, encoding="utf-8-sig")
    print(f"Row별 상세 결과가 '{output_filename}' 파일로 저장되었습니다.")

    print(f"\n--- [RNN_SEQ2SEQ 전체 요약 통계] ---")
    print("📊 [Macro-Averages (Row별 지표의 평균)]")
    macro_avg_cols = ['Overall-CER', 'Numeric-CER', 'Target-CER', 'Span-F1', 'Span-Precision', 'Span-Recall']
    macro_avg_metrics = metrics_df[macro_avg_cols].mean()
    print(macro_avg_metrics)
    
    total_tp = metrics_df['tp'].sum()
    total_fp = metrics_df['fp'].sum()
    total_fn = metrics_df['fn'].sum()
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    print("\n" + "---" + "\n")
    print("📈 [Micro-Averages (전체 Span 합산 기준 - F1 표준)]")
    print(f"  Span-F1 (Micro Avg): {micro_f1:.4f}")
    print(f"  (Span-Precision: {micro_precision:.4f})")
    print(f"  (Span-Recall: {micro_recall:.4f})")
    print("-------------------------------------------------")

    '''
    Row별 상세 결과가 'model_results_RNN_SEQ2SEQ_per_row.csv' 파일로 저장되었습니다.
        --- [RNN_SEQ2SEQ 전체 요약 통계] ---
        📊 [Macro-Averages (Row별 지표의 평균)]
        Overall-CER       0.033445
        Numeric-CER       0.195841
        Target-CER        0.427941
        Span-F1           0.191088
        Span-Precision    0.195328
        Span-Recall       0.192966
        dtype: float64

        ---

        📈 [Micro-Averages (전체 Span 합산 기준 - F1 표준)]
        Span-F1 (Micro Avg): 0.2068
        (Span-Precision: 0.1963)
        (Span-Recall: 0.2185)
        -------------------------------------------------
    '''
    
if __name__ == '__main__':
    main()