import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stdin.reconfigure(encoding='utf-8')
import io
import openai
import docx
import time
import fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uuid
import logging

# app = Flask(__name__)

sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')


# 로그 설정
logging.basicConfig(
    filename='chatbot_logs.log',
    level=logging.INFO,
    format='%(asctime)s %(session_id)s %(question)s %(answer)s'
)


# DOCX 파일을 섹션별로 읽어오기 함수
def read_docx_sections(file_path):
    try:
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

# # 가장 관련 있는 여러 섹션 찾기
# def find_relevant_sections(question, sections, top_n=3):
#     vectorizer = TfidfVectorizer()
#     vectors = vectorizer.fit_transform([question] + sections)
#     cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
#     most_relevant_indices = cosine_similarities.argsort()[-top_n:][::-1]
#     print(" === most_relevant_indices === ", most_relevant_indices)
#     return "\n\n".join([sections[i] for i in most_relevant_indices])

# 가장 관련 있는 여러 섹션 찾기
def find_relevant_sections(question, sections, top_n=3):
    # sections 리스트에서 NoneType이 아닌 요소만 유지
    valid_sections = [s for s in sections if s is not None]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([question] + valid_sections)
    cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    most_relevant_indices = cosine_similarities.argsort()[-top_n:][::-1]
    print(most_relevant_indices)
    return "\n\n".join([valid_sections[i] for i in most_relevant_indices])


# ChatGPT에게 메시지를 보내고 답변 받기
def send_message_to_chatgpt(user_question, context):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Answer strictly based on the provided document sections only."},
                {"role": "user", "content": f"Document Sections: {context}\n\nQuestion: {user_question}"},
            ],
            max_tokens=500,
            temperature=0.2,
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"ERROR: 모델을 호출할 수 없습니다. 이유: {e}")
        return None

if __name__ == '__main__':
    file_path = "dataset/data_sample.docx"
    sections = read_docx_sections(file_path) 
    
    if sections:
        print("안녕하세요. 오토앤 챗봇 앤돌이예요! 회사 생활 관련 궁금한 점이 생기면, 언제든지 앤돌이에게 물어보세요.")
        
        while True:
            user_input = input("사용자: ")
            
            if user_input.lower() in ["exit", "종료"]:
                print("대화를 종료합니다. 행복한 하루 보내세요!")
                break
            
            # 사용자 질문과 유사한 상위 섹션 찾기
            relevant_section = find_relevant_sections(user_input, sections)
            
            # ChatGPT에게 관련 섹션과 함께 질문 전송하여 응답 받기
            response = send_message_to_chatgpt(user_input, relevant_section)
            
            if response:
                print(f"앤돌이: {response}")
                
                # 로그 기록 시 extra 매개변수 사용
                session_id = str(uuid.uuid4())  # session_id를 문자열로 변환
                logging.info(
                    "",  # 메시지는 공백으로 남겨둠
                    extra={
                        'session_id': session_id,
                        'question': user_input,
                        'answer': response
                    }
                )
            else:
                print("앤돌이가 응답하지 않았습니다.")
    else:
        print("섹션을 읽어오는 데 실패했습니다.")