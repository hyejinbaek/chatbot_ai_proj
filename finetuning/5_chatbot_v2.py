import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# LangChain의 OpenAI LLM 객체를 생성합니다
fine_tuned_model = 'ft:gpt-4o-2024-08-06:auton::ASh95CCN'
llm = ChatOpenAI(model_name=fine_tuned_model, openai_api_key=openai_api_key)
# 시스템 메시지 정의
system_prompt = SystemMessage(content="You are a helpful assistant that answers questions based on internal company (오토앤) rules in a concise and courteous manner.")

# 대화 기록을 저장할 리스트
conversation_history = [system_prompt]

def chatbot_response(user_input):
    """
    사용자의 입력을 받아 OpenAI 모델을 통해 응답을 생성합니다.
    :param user_input: 사용자의 입력 텍스트
    :return: 생성된 응답 텍스트
    """
    try:
        # 사용자 메시지를 대화 기록에 추가
        conversation_history.append(HumanMessage(content=user_input))
        
        # invoke 메서드를 사용하여 응답 생성
        response = llm.invoke(conversation_history)
        
        # AI의 응답을 content에서 추출하여 텍스트로 반환
        response_text = response.content
        # 응답을 대화 기록에 추가
        conversation_history.append(response)
        
        return response_text
    except Exception as e:
        return f"오류가 발생했습니다: {e}"

def main():
    print("챗봇에 오신 것을 환영합니다! 끝내려면 '종료'라고 입력하세요.")
    while True:
        user_input = input("사용자: ")
        if user_input.lower() == "종료":
            print("챗봇을 종료합니다. 안녕히 가세요!")
            break

        # 사용자 입력에 대응하는 챗봇 응답 생성
        response = chatbot_response(user_input)
        print(f"챗봇: {response}")

if __name__ == "__main__":
    main()