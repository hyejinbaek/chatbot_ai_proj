import json
import re
from docx import Document

# 파일 경로 설정
# file_path = "dataset/data_sample_2.txt"
file_path = "dataset/data_sample.docx"

# 데이터를 담을 리스트
dataset = []

# docx 파일을 처리하는 함수
def process_docx(file_path):
    doc = Document(file_path)
    text = ""
    
    # 모든 문단을 순회하며 텍스트를 하나의 문자열로 결합
    for para in doc.paragraphs:
        text += para.text + "\n"
    
    # 추출된 텍스트를 처리하여 prompt와 completion 생성 (조항 단위)
    create_prompt_completion_for_docx(text)

# txt 파일을 처리하는 함수
def process_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    
    # 추출된 텍스트를 처리하여 prompt와 completion 생성 (문장 단위)
    create_prompt_completion_for_txt(text)

# 텍스트 정제 함수
def clean_text(text):
    # ①, ②, \n, \n\n 등의 불필요한 표현 제거
    text = re.sub(r"\d+\.", "", text)        # 번호 형태 제거 (예: ①, ②)
    text = re.sub(r"\n+", " ", text)         # 여러 줄 바꿈을 하나의 공백으로 대체
    text = re.sub(r"\s+", " ", text).strip() # 중복된 공백 제거 및 앞뒤 공백 제거
    return text

# DOCX 파일용 prompt와 completion 생성 함수 (조항 단위)
def create_prompt_completion_for_docx(text):
    # "제13조(인사위원회의 구성)"과 같이 조항을 기준으로 텍스트를 분리
    matches = re.findall(r"(제\d+조\((.+?)\))(.*?)((?=제\d+조\()|$)", text, re.DOTALL)
    
    for match in matches:
        full_clause = match[0].strip()       # 예: 제22조(근무형태)
        clause_topic = match[1].strip()      # 예: 근무형태
        clause_content = clean_text(match[2].strip())  # 조항 내용, 불필요한 표현 제거

        # 주제어를 활용하여 질문을 생성
        prompt = f"{clause_topic}에 대해 설명해줄 수 있어?"
        
        # 결과를 딕셔너리로 추가
        if clause_content:  # 빈 내용은 제외
            data = {"prompt": prompt, "completion": clause_content}
            dataset.append(data)

# TXT 파일용 prompt와 completion 생성 함수 (문장 단위)
def create_prompt_completion_for_txt(text):
    # 문장 단위로 분리
    sentences = re.split(r'[.!?]\s+', text)

    for sentence in sentences:
        sentence = clean_text(sentence.strip())
        if sentence:
            prompt = f"{sentence}에 대해 설명해줄 수 있어?"
            completion = sentence
            data = {"prompt": prompt, "completion": completion}
            dataset.append(data)

# 파일 형식에 따라 적절한 함수 호출
if file_path.endswith(".docx"):
    process_docx(file_path)
elif file_path.endswith(".txt"):
    process_txt(file_path)
else:
    print("지원하지 않는 파일 형식입니다.")

# 결과를 JSONL 파일로 저장
with open("../dataset/chatbot_dataset.jsonl", "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("Dataset 생성 완료:", dataset)
