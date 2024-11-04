# 이전 대화 기억 못함.
import openai
import json


openai.api_key = 

# JSON 파일에서 FAQ 데이터 로드
def load_faq(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 사용자 질문에 대한 답변 찾기
def get_answer(user_question):
    for item in faq_data:
        if user_question in item['question']:
            return item['answer']
    return None  # 답변을 찾지 못했을 경우 None 반환

# 최근 대화 기록을 저장할 리스트
conversation_history = []

def send_message_to_chatgpt(user_input):
    # 사용자 메시지 추가
    conversation_history.append({"role": "user", "content": user_input})
    
    # 대화 기록 중 최근 5개의 메시지만 가져옴
    recent_history = conversation_history[-10:]
    
    # 시스템 메시지와 최근 대화를 ChatGPT API에 전송
    conversation = [{"role": "system", "content": "You are a helpful assistant that answers questions based on internal company rules."}]
    conversation.extend(recent_history)

    try:
        # ChatGPT에게 메시지 전송
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # 사용할 모델 선택
            # model="gpt-4",
            messages=conversation,
            max_tokens=2048,
            temperature=0.7,
        )
        
        # 응답 텍스트 추출 및 반환
        response_text = response["choices"][0]["message"]["content"].strip()
        
        # Assistant의 응답을 대화 기록에 추가
        conversation_history.append({"role": "assistant", "content": response_text})
        
        return response_text

    except Exception as e:
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
    faq_file_path = "chatbot_qa_data.json"  # JSON 파일 경로
    faq_data = load_faq(faq_file_path)  # FAQ 데이터를 로드
    if faq_data:
        print("안녕하세요. 오토앤 챗봇 앤돌이예요! 회사 생활 관련 궁금한 점이 생기면, 언제든지 앤돌이에게 물어보세요.")
        
        while True:
            user_input = input("사용자: ")
            
            if user_input.lower() in ["exit", "종료"]:
                print("대화를 종료합니다. 행복한 하루 보내세요!")
                break
            
            # 사용자 질문에 대한 답변 찾기
            answer = get_answer(user_input)
            
            if answer:
                print(f"앤돌이: {answer}")
            else:
                # 사용자 질문을 ChatGPT에게 전송하고 응답 받기
                response = send_message_to_chatgpt(user_input)
                
                if response:
                    print(f"앤돌이: {response}")
                else:
                    print("앤돌이가 응답하지 않았습니다.")
            
            # '끝내겠니?' 질문 던지기
            if ask_to_end_conversation():
                break