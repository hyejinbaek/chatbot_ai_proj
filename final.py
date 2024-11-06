from flask import Flask, request, jsonify
import sys
import io
import openai
import docx
import time
import fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uuid
import logging

app = Flask(__name__)

sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')


# 로그 설정
logging.basicConfig(filename='chatbot_logs.log', level=logging.INFO, format='%(asctime)s %(session_id)s %(question)s %(answer)s')

# DOCX 파일을 섹션별로 읽어오기 함수
def read_docx_sections(file_path):
    doc = docx.Document(file_path)
    sections = []
    section_text = ""

    for para in doc.paragraphs:
        if para.text.strip():  # 빈 줄 무시
            section_text += para.text + "\n"
        else:
            if section_text:
                sections.append(section_text.strip())
                section_text = ""

    if section_text:  # 마지막 섹션 추가
        sections.append(section_text.strip())

    return sections

# PDF 파일을 섹션별로 읽어오기 함수
def read_pdf_sections(file_path):
    sections = []
    doc = fitz.open(file_path)
    
    for page in doc:
        text = page.get_text()
        sections.append(text.strip())
    
    return sections

# 여러 파일에서 섹션을 읽어와서 통합하는 함수
def read_documents(file_paths):
    all_sections = []
    
    for file_path in file_paths:
        if file_path.lower().endswith('.docx'):
            all_sections.extend(read_docx_sections(file_path))
        elif file_path.lower().endswith('.pdf'):
            all_sections.extend(read_pdf_sections(file_path))
        else:
            print(f"지원하지 않는 파일 형식: {file_path}")

    return all_sections

# 가장 관련 있는 섹션 찾기
def find_relevant_section(question, sections):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([question] + sections)
    cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    most_relevant_idx = cosine_similarities.argmax()
    return sections[most_relevant_idx]

# 챗봇 함수 정의
def ask_bot(question, context):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "당신은 회사 규정에 기반하여 질문에 답변하는 어시스턴트입니다. 다음 내용을 참조하십시오:"},
            {"role": "user", "content": context},
            {"role": "user", "content": question},
        ],
        max_tokens=300,
        temperature=0.2,
    )
    return response['choices'][0]['message']['content']

# 사용자 세션별 로그 기록
def log_interaction(session_id, question, answer):
    logging.info(f'{session_id} {question} {answer}')

# API 엔드포인트: 질문 요청 처리
@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    user_question = data.get('question')
    file_paths = data.get('file_paths')
    
    if not user_question or not file_paths:
        return jsonify({'error': '질문과 파일 경로가 필요합니다.'}), 400

    # 고유한 세션 ID 생성
    session_id = str(uuid.uuid4())

    # 문서에서 섹션 불러오기
    sections = read_documents(file_paths)
    
    # 관련 섹션 찾기 및 응답 생성
    try:
        relevant_section = find_relevant_section(user_question, sections)
        start_time = time.time()
        bot_answer = ask_bot(user_question, relevant_section)
        end_time = time.time()
        
        inference_time = end_time - start_time
        response_data = {
            'session_id': session_id,
            'question': user_question,
            'answer': bot_answer,
            'inference_time': inference_time
        }

        # 로그 기록
        log_interaction(session_id, user_question, bot_answer)

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
