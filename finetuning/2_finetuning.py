import openai
import os
import sys
import io
import time

sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Chat 형식 데이터 파일 업로드
training_file = openai.File.create(
    file=open("./complete/dataset_type_change.jsonl", "rb"),  
    purpose="fine-tune"
)

# Fine-Tuning 작업 생성
fine_tune_job = openai.FineTuningJob.create(
    training_file=training_file['id'],  
    hyperparameters={
        "n_epochs": 10,
    },
    model="gpt-4o-2024-08-06"
    # 기존 fine-tuning한 모델 id 입력하면 됨.
    # model = "ft:gpt-4o-mini-2024-07-18:auton::ARBHi5t9"
)

# Retrieve and print logs periodically
def print_fine_tune_logs(job_id):
    while True:
        result = openai.FineTuningJob.retrieve(id=job_id)
        status = result['status']
        # Print the current status of the fine-tuning job
        print(f"Status: {status}")
        
        # Print logs if the job is still running
        if status in ['running', 'pending']:
            logs = openai.FineTuningJobLogs.retrieve(id=job_id)
            for line in logs:
                print(line['message'])
                
            # Sleep for a while before retrieving logs again
            time.sleep(30)
        else:
            print("Fine-tuning job completed.")
            break

# Start printing logs
print_fine_tune_logs(fine_tune_job['id'])