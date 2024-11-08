import json

# 기존 데이터 파일을 읽기
with open("../dataset/chatbot_dataset.jsonl", "r", encoding="utf-8") as file:
    lines = file.readlines()

# 새 파일에 Chat 형식으로 데이터 저장
with open("data_prepared_chat.jsonl", "w", encoding="utf-8") as file:
    for line in lines:
        data = json.loads(line)
        chat_format = {
            "messages": [
                {"role": "user", "content": data["prompt"]},
                {"role": "assistant", "content": data["completion"]}
            ]
        }
        file.write(json.dumps(chat_format, ensure_ascii=False) + "\n")