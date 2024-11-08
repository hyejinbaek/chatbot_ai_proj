import openai


# 파인튜닝된 모델 ID (파인튜닝이 완료되면 확인 가능)
FINE_TUNED_MODEL = "ft:gpt-4o-mini-2024-07-18:auton::ARBHi5t9"

# 최근 대화 기록을 저장할 리스트
conversation_history = []

# 챗봇 응답 함수
def chatbot_response(question):# 사용자 메시지 추가
    conversation_history.append({"role": "user", "content": question})
    
    # 대화 기록 중 최근 5개의 메시지만 가져옴
    recent_history = conversation_history[-10:]
    
    # 시스템 메시지와 최근 대화를 ChatGPT API에 전송
    conversation = [{"role": "system", "content": "You are a helpful assistant that answers questions based on internal company(오토앤) rules."}]
    conversation.extend(recent_history)

    try:
        # ChatGPT에게 메시지 전송
        response = openai.ChatCompletion.create(
            model=FINE_TUNED_MODEL,  # 사용할 모델 선택
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

# 챗봇 사용 예시
while True:
    question = input("질문을 입력하세요: ")
    if question.lower() in ["exit", "종료", "끝"]:
        print("대화를 종료합니다. 행복한 하루 보내세요!")
        break
    answer = chatbot_response(question)
    print(f"챗봇 응답: {answer}")
    
    # '끝내겠니?' 질문 던지기
    if ask_to_end_conversation():
        break
