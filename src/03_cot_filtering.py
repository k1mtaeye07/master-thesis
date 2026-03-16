from datasets import load_dataset

DATASET_PATH = "/workspace/data/cot_11900_dataset.jsonl"

dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

print(f"필터링 전 샘플 수: {len(dataset)}")
dataset = dataset.filter(lambda x: x['status'] != 'OK')
print(f"필터링 후 샘플 수 (Status=OK): {len(dataset)}")

dataset.to_json("/workspace/data/cot_11900_dataset_filtered.jsonl", force_ascii=False, orient = 'records', indent=4)