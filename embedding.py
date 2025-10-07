import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path


# 1. BGE-M3 모델 로드
MODEL_NAME = "BAAI/bge-m3"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def get_embedding(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state[:, 0, :]  # CLS 토큰
    return embeddings[0].cpu().numpy().astype("float32")


# 2. JSON 로드
input_path = Path("parsed_docx.json")
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

#list / dict 모두 대응
if isinstance(data, list):
    documents = data
elif isinstance(data, dict):
    documents = data.get("documents", [])
else:
    raise ValueError("지원하지 않는 JSON 구조입니다.")


# 3. 임베딩 생성
vectors = np.array([get_embedding(doc) for doc in documents])

print("임베딩 shape:", vectors.shape)


# 4. 임베딩 + 원문 저장
output_dir = Path("embeddings")
output_dir.mkdir(exist_ok=True)

#넘파이 파일로 저장
np.save(output_dir / "doxc_embeddings.npy", vectors)

#JSON으로 저장 (텍스트 + 벡터 매핑)
embedding_data = [
    {"text": doc, "vector": vec.tolist()} 
    for doc, vec in zip(documents, vectors)
]

with open(output_dir / "docx_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(embedding_data, f, ensure_ascii=False, indent=2)

print("✅ 임베딩 저장 완료: embeddings/doc_embeddings.npy, embeddings/doc_embeddings.json")
