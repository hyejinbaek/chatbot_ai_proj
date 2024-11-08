import openai
from flask import Flask, request, jsonify

app = Flask(__name__)


# 파인튜닝된 모델 ID (파인튜닝이 완료되면 확인 가능)
FINE_TUNED_MODEL = "ft:gpt-4o-mini-2024-07-18:auton::ARBHi5t9"

# 최근 대화 기록을 저장할 리스트
conversation_history = []

# 챗봇 응답 함수
def chatbot_response(question):
    # 사용자 메시지 추가
    conversation_history.append({"role": "user", "content": question})
    
    # 대화 기록 중 최근 10개의 메시지만 가져옴
    recent_history = conversation_history[-10:]
    
    # 시스템 메시지와 최근 대화를 ChatGPT API에 전송
    conversation = [{"role": "system", "content": "You are a helpful assistant that answers questions based on internal company(오토앤) rules."}]
    conversation.extend(recent_history)

    try:
        # ChatGPT에게 메시지 전송
        response = openai.ChatCompletion.create(
            model=FINE_TUNED_MODEL,
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
