# txt로 만든 파일을 임베딩 미리 계산하여 저장(시간 단축을 위함)
import openai
import json
import numpy as np

openai.api_key = 

# FAQ 파일 경로
faq_file_path = "test_sample.json"
embedded_faq_file_path = "embedded_faq_data.json"

# FAQ 데이터 로드
def load_faq(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 문장을 임베딩 벡터로 변환
def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

# FAQ 데이터를 임베딩하여 JSON 파일로 저장
def embed_and_save_faq_data(faq_data, output_file):
    for item in faq_data:
        item['embedding'] = get_embedding(item['question'])
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(faq_data, f, ensure_ascii=False, indent=4)

# FAQ 데이터 임베딩 후 저장
faq_data = load_faq(faq_file_path)
embed_and_save_faq_data(faq_data, embedded_faq_file_path)
