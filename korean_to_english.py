# 폴더명 한국어 -> 영어로 변경
import os
from googletrans import Translator

# 번역기 객체 생성
translator = Translator()

# 파일이 저장된 디렉토리 경로
directory = 'dataset/241113'  # 파일들이 있는 디렉토리로 변경하세요.

# 디렉토리 내 모든 파일 및 폴더를 재귀적으로 탐색
for foldername, subfolders, filenames in os.walk(directory):
    for file in filenames:
        # 파일 경로 추출
        file_path = os.path.join(foldername, file)
        
        # 파일 이름에서 확장자를 제외한 이름 추출
        filename, extension = os.path.splitext(file)
        
        # 한국어 파일 이름을 영어로 번역
        translated_name = translator.translate(filename, src='ko', dest='en').text
        
        # 새 파일 이름 생성
        new_file_name = translated_name + extension
        
        # 파일 이름 변경
        new_file_path = os.path.join(foldername, new_file_name)
        os.rename(file_path, new_file_path)

        print(f'{file_path} -> {new_file_path}')
