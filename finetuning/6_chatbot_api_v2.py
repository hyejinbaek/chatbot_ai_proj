import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

app = Flask(__name__)

# 환경 변수 로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# 파인튜닝된 모델 ID
FINE_TUNED_MODEL = "ft:gpt-4o-mini-2024-07-18:auton::ARBHi5t9"
llm = ChatOpenAI(model_name=FINE_TUNED_MODEL, openai_api_key=openai_api_key)

# 시스템 메시지 정의 및 초기화
system_prompt = SystemMessage(content="You are a helpful assistant that answers questions based on internal company (오토앤) rules.")
conversation_history = [system_prompt]

# 챗봇 응답 함수
def chatbot_response(question):
    try:
        # 사용자 입력을 대화에 추가
        conversation_history.append(HumanMessage(content=question))
        
        # LangChain 모델 호출
        response = llm.invoke(conversation_history)
        
        # 응답 텍스트 추출
        response_text = response.content
        # AI 응답을 대화 기록에 추가
        conversation_history.append(response)
        
        return response_text

    except Exception as e:
        print(f"ERROR: 모델 호출 오류 - {e}")
        return "An error occurred while processing your request."

# 기본 엔드포인트
@app.route('/')
def index():
    return "API is running. Use /ask to interact with the chatbot.", 200

# 챗봇 API 엔드포인트
@app.route('/ask', methods=['POST'])
def ask_chatbot():
    data = request.get_json()
    question = data.get("question")
    
    if not question:
        return jsonify({"error": "No question provided"}), 400

    # 챗봇 응답 가져오기
    answer = chatbot_response(question)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
