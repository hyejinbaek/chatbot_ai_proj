import boto3
from botocore.exceptions import ClientError
import json
from datetime import datetime
import uuid  # 세션 ID 생성용

# AWS Bedrock 클라이언트 생성
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# JSON 파일에서 FAQ 데이터 로드
def load_faq(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 사용자 질문 및 모델 답변 로그 저장 (타임스탬프와 세션 ID 추가)
def log_user_interaction(session_id, question, answer, log_file_path="user_interactions_log.json"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 현재 시간 타임스탬프

    # 로그 데이터 구조
    log_entry = {
        "timestamp": timestamp,
        "session_id": session_id,
        "question": question,
        "answer": answer
    }

    # 기존 로그 파일이 있다면 읽어오기
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            logs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logs = []

    # 새로운 로그 추가
    logs.append(log_entry)

    # 로그 파일에 다시 저장
    with open(log_file_path, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=4)

# 사용자 질문에 대한 답변 찾기
def get_answer(user_question):
    for item in faq_data:
        if user_question in item['question']:
            return item['answer']
    return None  # 답변을 찾지 못했을 경우 None 반환

def send_message_to_claude(user_input):
    # Claude에게 보낼 메시지 구성
    user_message = f"You are an AI customer success agent. Answer based on the following FAQ:\n{json.dumps(faq_data, ensure_ascii=False)}\n\nBEGIN DIALOGUE\n\nHuman: {user_input}\n"
    
    conversation = [
        {
            "role": "user",
            "content": [{"text": user_message}],
        }
    ]
    
    try:
        # Claude에게 메시지 전송
        response = client.converse(
            modelId="anthropic.claude-v2",
            messages=conversation,
            inferenceConfig={
                "maxTokens": 2048,
                "stopSequences": ["\n\nHuman:", "\n\nAssistant:"],
                "temperature": 0,
                "topP": 1
            },
            additionalModelRequestFields={"top_k": 250}
        )
        
        # 응답 텍스트 추출
        response_text = response["output"]["message"]["content"][0]["text"].strip()

        # 불필요한 부분을 제거하고, AI의 답변만 남기기
        if "Assistant:" in response_text:
            response_text = response_text.split("Assistant:")[-1].strip()

        # 불필요한 "Human:" 태그 및 질문을 제거
        if "Human:" in response_text:
            response_text = response_text.split("Human:")[-1].strip()

        return response_text

    except (ClientError, Exception) as e:
        print(f"ERROR: 모델을 호출할 수 없습니다. 이유: {e}")
        return None


# 사용자와 상호작용
def ask_to_end_conversation():
    """질문을 던져서 사용자가 끝내겠다고 하면 종료"""
    while True:
        end_conversation_input = input("앤돌이: 질문을 다 하셨나요? (예/아니요): ")
        if end_conversation_input.lower() in ["예", "네", "yes", "응", "어"]:
            print("대화를 종료합니다. 행복한 하루 보내세요!")
            return True
        elif end_conversation_input.lower() in ["아니", "아니요", "no", "노", "ㄴㄴ"]:
            return False
        else:
            print("잘못된 입력입니다. 예/아니요로 답해주세요.")

if __name__ == "__main__":
    # FAQ 내용 로드
    # faq_file_path = "chatbot_qa_data.json"
    faq_file_path = "chatbot_data.json"
    faq_data = load_faq(faq_file_path)  # FAQ 데이터를 로드
    if faq_data:
        print("안녕하세요. 오토앤 챗봇 앤돌이예요! 회사 생활 관련 궁금한 점이 생기면, 언제든지 앤돌이에게 물어보세요.")
        
        while True:
            # 새로운 사용자 세션 ID 생성
            session_id = str(uuid.uuid4())
            
            user_input = input("사용자: ")
            
            if user_input.lower() in ["exit", "종료"]:
                print("대화를 종료합니다. 행복한 하루 보내세요!")
                break
            
            # 사용자 질문에 대한 답변 찾기
            answer = get_answer(user_input)
            
            if answer:
                print(f"앤돌이: {answer}")
                # 사용자 질문과 모델 답변 로그 저장
                log_user_interaction(session_id, user_input, answer)
            else:
                # 사용자 질문을 Claude에게 전송하고 응답 받기
                response = send_message_to_claude(user_input)
                
                if response:
                    print(f"앤돌이: {response}")
                    # 사용자 질문과 Claude 모델 답변을 로그에 저장
                    log_user_interaction(session_id, user_input, response)
                else:
                    print("앤돌이가 응답하지 않았습니다.")
            
            # '끝내겠니?' 질문 던지기
            if ask_to_end_conversation():
                break
