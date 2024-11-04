# data preprocessing
# C:\Users\hyejinbaek\Desktop\100\code\chatbot\data_sample.pdf

import fitz
import argparse
import os

def pdf_to_text(pdf_input_path):
    doc = fitz.open(pdf_input_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text
    
def manage_data():
    parser = argparse.ArgumentParser(description='face embedding and comparison')
    args = parser.parse_args()

    while True:
        file_input = input("input file type?(0 or 1) : ")
        # 0 : pdf / 1 : docx, txt
        if file_input in ["0", "1"]:
            args.regist = int(file_input)
            if args.regist == 0:
                pdf_input_path = input("input pdf path : ")
                text = pdf_to_text(pdf_input_path)
                print(text)
                break
            else:
                docx_input_path = input("docx input path : ")

def user_data():
    text = pdf_to_text('data_sample.pdf')
    # print(text)
    import re
    import json

    # 비정형 데이터를 텍스트로 입력합니다.
    raw_text = text

    # 패턴을 정의합니다.
    # 각 조문의 번호와 제목을 패턴으로 사용
    chapter_pattern = r"(제\s*\d+\s*장|제\s*\d+\s*조)"
    item_pattern = r"\d+\.\s*(.*)"

    # 질문과 답변을 저장할 리스트
    chatbot_data = []

    # 조문과 내용을 검색합니다.
    # chapter_info에 조문 정보를 담아 사용할 수 있도록 합니다.
    chapter_info = re.findall(chapter_pattern, raw_text)

    # 각 장/조문에서 항목을 찾아 질문과 답변 생성
    for chapter in chapter_info:
        # 해당 장/조문에 대한 항목 내용 추출
        items = re.findall(item_pattern, raw_text[raw_text.find(chapter):])

        # 항목이 있으면 질문과 답변을 생성
        for index, item in enumerate(items):
            # 질문 생성: 각 항목에 대한 고유한 질문
            question = f"{chapter}의 항목 {index + 1}에 대해 설명해줄 수 있어?"
            answer = item.strip()
            chatbot_data.append({"question": question, "answer": answer})

    # # JSON 형태로 변환
    # with open("chatbot_data.json", "w", encoding="utf-8") as json_file:
    #     json.dump(chatbot_data, json_file, ensure_ascii=False, indent=4)

    # print("챗봇 데이터셋이 chatbot_data.json에 저장되었습니다.")

    
    return
            
if __name__ == "__main__":
    # manage_data()
    user_data()