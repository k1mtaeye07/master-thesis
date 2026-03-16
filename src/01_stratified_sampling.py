import pandas as pd
from sklearn.model_selection import train_test_split

# --- 1. 설정 ---
INPUT_FILE = "/workspace/data/train_set_preprocessed.csv"
OUTPUT_FILE = "/workspace/data/golden_11900_stratified_samples.csv"
TARGET_SAMPLE_SIZE = 11900  # 119,902건 중 약 10%
STRATIFY_COLUMN = "depth1"  # 계층화 기준이 되는 컬럼

def main():
    print(f"--- 1. '{INPUT_FILE}' 로드 중... ---")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"오류: '{INPUT_FILE}'을(를) 찾을 수 없습니다. 경로를 확인하세요.")
        return
    except Exception as e:
        print(f"파일 로드 중 오류 발생: {e}")
        return

    total_rows = len(df)
    if total_rows == 0:
        print("오류: 파일이 비어있습니다.")
        return

    # --- 2. 계층적 샘플링 계산 ---
    # train_test_split을 활용하여 계층적 '샘플'을 추출합니다.
    # (TARGET_SAMPLE_SIZE / total_rows) 비율만큼 'train' 세트로 뽑아냅니다.
    sample_ratio = TARGET_SAMPLE_SIZE / total_rows
    
    # [중요] 
    # stratify=df[STRATIFY_COLUMN]
    # 이 옵션이 'depth1'의 분포를 유지하면서 샘플을 추출하는 핵심입니다.
    print(f"--- 2. 전체 {total_rows}개 중 {TARGET_SAMPLE_SIZE}개 ({sample_ratio:.2%}) 계층적 샘플링 시작... ---")
    print(f"       기준 컬럼: '{STRATIFY_COLUMN}'")
    
    try:
        # train_size를 기준으로 샘플링
        stratified_sample_df, _ = train_test_split(
            df,
            train_size=TARGET_SAMPLE_SIZE,
            stratify=df[STRATIFY_COLUMN],
            random_state=42  # 재현성을 위한 고정 시드
        )
    except ValueError as e:
        print(f"\n[오류] 계층적 샘플링 실패. 'depth1'의 일부 카테고리 샘플 수가 1개일 수 있습니다.")
        print(f"오류 메시지: {e}")
        print("샘플링을 중단합니다.")
        return

    # --- 3. 결과 검증 및 저장 ---
    print(f"--- 3. 샘플링 완료. 총 {len(stratified_sample_df)}개 샘플 생성. ---")
    
    # 원본 분포와 샘플 분포 비교 (상위 5개)
    #print("\n[샘플링 분포 검증 (상위 5개)]")
    #original_dist = df[STRATIFY_COLUMN].value_counts(normalize=True).head(5)
    #sample_dist = stratified_sample_df[STRATIFY_COLUMN].value_counts(normalize=True).head(5)
    print("\n[샘플링 분포 검증 전체]")
    original_dist = df[STRATIFY_COLUMN].value_counts(normalize=True)
    sample_dist = stratified_sample_df[STRATIFY_COLUMN].value_counts(normalize=True)
    
    comparison_df = pd.DataFrame({
        'Original_Distribution': original_dist,
        'Sample_Distribution': sample_dist
    })
    print(comparison_df)

    # 파일로 저장
    # stratified_sample_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    # print(f"\n--- 4. 성공: '{OUTPUT_FILE}' 파일로 저장되었습니다. ---")

if __name__ == "__main__":
    main()