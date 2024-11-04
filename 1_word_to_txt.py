# word 파일 -> txt 파일 형식 변환
import docx
import json

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    content = []
    current_title = ""

    for para in doc.paragraphs:
        text = para.text.strip()
        
        if para.style.name.startswith('Heading') and text:  # 제목인 경우
            current_title = text
            content.append({"type": "title", "text": current_title})
        elif text.startswith(("1.", "2.", "①", "②")):  # 번호 매긴 목록인 경우
            content.append({"type": "list", "text": text})
        elif text:  # 일반 텍스트
            content.append({"type": "paragraph", "text": text})

    return content

def convert_to_json(extracted_content):
    faq_data = []
    current_title = ""

    for item in extracted_content:
        if item["type"] == "title":  # 제목을 새로운 질문의 기본으로 설정
            current_title = item["text"]
            # 제목 자체를 질문으로 추가
            faq_data.append({
                "question": f"{current_title}에 대해 설명해줄 수 있어?",
                "answer": current_title
            })
        elif item["type"] == "paragraph":  # 일반 텍스트는 제목을 포함한 답변에 추가
            faq_data.append({
                "question": f"{current_title}의 내용에 대해 설명해줄 수 있어?",
                "answer": item["text"]
            })
        elif item["type"] == "list":  # 목록 항목을 개별 질문으로 추가
            faq_data.append({
                "question": f"{current_title}의 항목 {item['text']}에 대해 설명해줄 수 있어?",
                "answer": item["text"]
            })

    return faq_data

# 예시 파일 경로
file_path = "data_sample.docx"
extracted_content = extract_text_from_docx(file_path)

# 추출된 내용을 JSON 형식으로 변환
faq_json_data = convert_to_json(extracted_content)

# JSON 파일로 저장
with open('test_sample.json', 'w', encoding='utf-8') as json_file:
    json.dump(faq_json_data, json_file, ensure_ascii=False, indent=4)

print("FAQ 데이터를 JSON 파일로 저장했습니다.")


