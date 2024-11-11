import openai
import os
import sys
import io
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')


# OpenAI API 키 설정

# Chat 형식 데이터 파일 업로드
training_file = openai.File.create(
    # 데이터 jsonl 파일 변경하면서 추가 fine-tuning 진행
    file=open("complete/chat_data_241111.jsonl", "rb"),  
    purpose="fine-tune"
)

# Fine-Tuning 작업 생성
fine_tune_job = openai.FineTuningJob.create(
    training_file=training_file['id'],  
    hyperparameters={
        "n_epochs": 3,  
        
    },
    # model="gpt-4o-mini-2024-07-18"
    
    # 기존 fine-tuning한 모델 id 입력하면 됨.
    model = "ft:gpt-4o-mini-2024-07-18:auton::ARBHi5t9"
)