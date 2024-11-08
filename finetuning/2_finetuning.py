import openai
import os
import sys
import io
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')



# Chat 형식 데이터 파일 업로드
training_file = openai.File.create(
    file=open("data_prepared_chat.jsonl", "rb"),  
    purpose="fine-tune"
)

# Fine-Tuning 작업 생성
fine_tune_job = openai.FineTuningJob.create(
    training_file=training_file['id'],  
    hyperparameters={
        "n_epochs": 3,  
        
    },
    model="gpt-4o-mini-2024-07-18"  
)