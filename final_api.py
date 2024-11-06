from flask import Flask, request, jsonify
import openai
import docx
import fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uuid
import logging

app = Flask(__name__)

# OpenAI API 키 설정

# 로그 설정
#logging.basicConfig(
#    filename='chatbot_logs.log',
#    level=logging.INFO,
#    format='%(question)s %(answer)s'
#)

# DOCX 파일을 섹션별로 읽어오기 함수
def read_docx_sections(file_path):
    try:
        doc = docx.Document(file_path)
        sections = []
        section_text = ""

        for para in doc.paragraphs:
            if para.text.strip():
                section_text += para.text + "\n"
            else:
                if section_text:
                    sections.append(section_text.strip())
                    section_text = ""
        if section_text:
            sections.append(section_text.strip())

        return sections
    except Exception as e:
        print(f"DOCX 파일을 읽는 중 오류 발생: {e}")
        return []

# PDF 파일을 섹션별로 읽어오기 함수
def read_pdf_sections(file_path):
    sections = []
    try:
        doc = fitz.open(file_path)
        
        for page in doc:
            text = page.get_text()
            sections.append(text.strip())
    
    except Exception as e:
        print(f"PDF 파일을 읽는 중 오류 발생: {e}")

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

# 가장 관련 있는 여러 섹션 찾기
def find_relevant_sections(question, sections, top_n=3):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([question] + sections)
    cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    most_relevant_indices = cosine_similarities.argsort()[-top_n:][::-1]
    return "\n\n".join([sections[i] for i in most_relevant_indices])

# ChatGPT에게 메시지를 보내고 답변 받기
def send_message_to_chatgpt(user_question, context):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Answer strictly based on the provided document sections only."},
                {"role": "user", "content": f"Document Sections: {context}\n\nQuestion: {user_question}"},
            ],
            max_tokens=300,
            temperature=0.2,
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"ERROR: 모델을 호출할 수 없습니다. 이유: {e}")
        return None


# basic api path
@app.route('/')
def index():
    return "API is running. Use /ask to interact with the chatbot.", 200


# Flask API 엔드포인트
@app.route('/ask', methods=['POST'])
def ask_chatbot():
    data = request.json
    user_question = data.get('question', '')
    
    # 여러 파일 경로를 리스트로 받음
    file_paths = ["dataset/data_sample.docx"]  # 파일 경로 설정
    sections = read_documents(file_paths)
    
    if sections:
        # 사용자 질문과 유사한 상위 섹션 찾기
        relevant_section = find_relevant_sections(user_question, sections)
        
        # ChatGPT에게 관련 섹션과 함께 질문 전송하여 응답 받기
        response = send_message_to_chatgpt(user_question, relevant_section)
        
        if response:
            # 세션 ID 생성
            #session_id = str(uuid.uuid4())
            
            # 로그 기록
            #logging.info(f'q : {question} a : {answer}')
            # 사용자 세션별 로그 기록
            
            # 응답 반환
            return jsonify({"answer": response})
        else:
            return jsonify({"error": "챗봇 응답 오류 발생"}), 500
    else:
        return jsonify({"error": "문서를 읽어오지 못했습니다."}), 500

if __name__ == '__main__':
    print("success")
    app.run(host='0.0.0.0', port=5000)

