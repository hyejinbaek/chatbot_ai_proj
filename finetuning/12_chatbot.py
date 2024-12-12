# openai = 1.57.3
# vector store값 불러온 챗봇

import os
import json
import uuid
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.document_loaders import PyMuPDFLoader, PyPDFLoader, TextLoader
from flask import Flask, request, jsonify, session
import unicodedata
from langchain.schema import Document
from docx import Document as DocxDocument
from datetime import datetime
import pandas as pd
import threading

app = Flask(__name__)

load_dotenv()
app.secret_key = os.getenv('FLASK_SECRET_KEY')
openai_api_key = os.getenv("OPENAI_API_KEY")
fine_tuned_model = 'ft:gpt-3.5-turbo-1106:auton::AXJCxOp9'

log_lock = threading.Lock()

def save_conversation_log(session_id, question, answer):
    with log_lock:  # 파일 접근을 동기화
        if session_id not in conversation_logs:
            conversation_logs[session_id] = []
        conversation_logs[session_id].append({
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer
        })

        conversation_log_path = get_conversation_log_path()
        with open(conversation_log_path, 'w', encoding='utf-8') as f:
            json.dump(conversation_logs, f, ensure_ascii=False, indent=4)

# 대화 기록 저장 경로
def get_conversation_log_path():
    today_date = datetime.now().strftime("%Y%m%d")
    return f"./log/conversation_logs_{today_date}.json"

# 대화 기록 로드 또는 초기화
def load_conversation_logs():
    conversation_log_path = get_conversation_log_path()
    if os.path.exists(conversation_log_path):
        with open(conversation_log_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

conversation_logs = load_conversation_logs()

# 새로운 세션 ID 생성
def generate_session_id():
    return str(uuid.uuid4())


# 이전 대화 기록 불러오기
def get_recent_context(session_id, max_history=5):
    if session_id in conversation_logs:
        return conversation_logs[session_id][-max_history:]
    return []

def format_context(recent_context):
    context = ""
    for idx, entry in enumerate(recent_context):
        context += f"[대화 {idx+1}] 질문: {entry['question']}, 답변: {entry['answer']}\n"
    return context



# 벡터 스토어 경로 설정
vectorstore_path = "./vector/faiss_index"

# 벡터 스토어 로드
if os.path.exists(vectorstore_path):
    print("벡터 스토어 로드 중...")
    embeddings = OpenAIEmbeddings()  # embeddings 객체 필요
    vectorstore = FAISS.load_local(
        vectorstore_path, 
        embeddings=embeddings, 
        allow_dangerous_deserialization=True  # 안전한 파일임을 확신하는 경우 사용
    )
else:
    raise FileNotFoundError(f"벡터 스토어 파일이 {vectorstore_path} 경로에 없습니다. 확인 후 다시 시도하세요.")

# Retriever 생성
retriever = vectorstore.as_retriever(k=5)

    

prompt = PromptTemplate.from_template(
        """
        이 AI 챗봇은 오토앤 회사의 내부 정책과 절차에 대한 질문에 답변하기 위해 설계되었습니다.
        직원(사용자)들이 회사 규정, 휴가 제도(연차), 복리후생, 또는 기타 회사 관련 사항에 대해 질문하면, 
        주어진 맥락을 기반으로 정확하고 신뢰성 있는 답변을 제공하세요.

        **프롬프트 사용 규칙**:
        1. 사용자가 간단한 인사를 하면, 정중하고 따뜻한 어조로 응답합니다. 예를 들어, 
        - "안녕?" → "안녕하세요! 오늘도 좋은 하루 되세요."
        - "고마워" → "천만에요! 도움이 되었다니 기쁩니다." 외에도 다양한 표현으로 따뜻하고 친절하게 응답하세요.
        2. 사용자가 오토앤의 특정 정책이나 절차를 물어볼 경우:
        - 질문과 관련된 정보가 컨텍스트에 포함되어 있다면 이를 활용해 명확하고 간결하게 설명하세요.
        - 예를 들어: "오토앤 휴가 제도는 무엇인가요?"라는 질문에 대해 컨텍스트에 정보가 있다면,
            "오토앤의 휴가 제도는 연차, 병가, 특별 휴가로 구성되어 있습니다. 추가적으로 궁금한 사항이 있다면 알려주세요!"와 같이 응답합니다.
        - 사용자의 질문이 명확하지 않거나 키워드가 부족한 경우, 구체적으로 질문할 수 있도록 친절히 안내하세요.
        - 예를 들어 : "오토앤에서 일할 때 규정이 있나요?"라고 물었다면,
            "오토앤에서 적용되는 규정을 말씀하시는 건가요? 예를 들어, 근무 시간, 사내 동호회, 혹은 연차와 관련된 규정을 구체적으로 질문해 주시면 더 정확한 답변을 드릴 수 있습니다."
        - 사용자가 질문을 더 명확히 할 수 있도록 다음과 같이 추가적인 예시를 제시하세요 : 
            "근무 시간 관련 규정인가요?", "사내 동호회에 대한 정보가 필요하신가요?"
        3. 질문과 관련된 정보가 컨텍스트에 없거나 모호할 경우:
        - 친절한 태도로 사용자에게 직접 경영지원본부 또는 적절한 부서로 문의할 것을 안내합니다.
        - 예를 들어: "이 부분에 대한 정확한 정보를 제공하기 어렵습니다. 경영지원본부 또는 관련 부서에 문의해 주시면 도움을 받으실 수 있습니다."
        4. 응답 시 사용자가 이해하기 쉬운 언어를 사용하며, 과도한 전문 용어나 복잡한 설명을 피합니다.
        5. 한국어로만 응답하세요.
        6. 사용자가 너의(AI챗봇) 존재를 묻는다면 친절하게 설명해주세요. 예를 들어,
        - 오토앤에 대해 궁금한 점이 있으시군요! 회사의 정책, 절차, 복리후생, 휴가 제도 등 다양한 정보를 제공할 수 있습니다. 어떤 특정한 정보가 필요하신가요? 예를 들어, "휴가 제도에 대해 알고 싶어요" 또는 "복리후생이 어떤 게 있나요?"와 같이 구체적으로 질문해 주시면 더 정확한 답변을 드릴 수 있습니다.
        와 같이 친절하게 다양한 표현으로 설명하세요.
        7. 말투를 다정하고 따뜻하게 이모티콘도 쓰면서 답변하세요.
        
        **주의사항**:
        - 응답은 너무 길거나 복잡하지 않게 간결하고 친절하게 작성합니다.
        - 오답을 줄이기 위해 컨텍스트 정보가 명확하지 않으면 반드시 문의를 유도하세요.
        - 모든 응답은 따뜻하고 협조적인 태도로 제공되어야 합니다.

        #Question: 
        {question} 
        #Context: 
        {context} 

        #Answer:""" 
)

llm = ChatOpenAI(model_name=fine_tuned_model, temperature=0)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def ask_question(session_id, question):
    recent_context = get_recent_context(session_id)
    formatted_context = format_context(recent_context)
    print(f"Context found: {formatted_context}")

    full_input = f"""
    # 최근 대화 기록:
    {formatted_context}

    # 질문: 
    {question} 
    """
    try:
        # 검색 결과 가져오기
        results = retriever.invoke(question)
        print("검색 결과:", results)
        
        # 결과를 기반으로 LLM 호출
        response = chain.invoke(full_input)
        print("프롬프트 입력값:", full_input)
        save_conversation_log(session_id, question, response)
        return response
    except Exception as e:
        print(f"ERROR: 모델 호출 오류 - {e}")
        return "죄송합니다. 답변을 생성하는 중 오류가 발생했습니다."

@app.route('/')
def index():
    return "API is running. Use /ask to interact with the chatbot.", 200

    
@app.route('/ask', methods=['POST'])
def ask_chatbot():
    data = request.get_json()
    question = data.get("question")
        
    # 디버깅: 세션 ID 확인
    if 'session_id' not in session:
        print("새로운 세션 ID 생성")
        session['session_id'] = str(uuid.uuid4())  # 고유한 세션 ID 생성

    session_id = session['session_id']
    print(f"Session ID: {session_id}")  # 디버깅 로그
        
    if not question:
        return jsonify({"error": "No question provided"}), 400

    answer = ask_question(session_id, question)

    return jsonify({
        "answer": answer,
        "session_id": session_id
    })
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
