# file-classification-llm
한국어 LLM 모델을 활용한 카테고리 분류 시스템

## 전제 조건
1) Cuda 환경을 지원하는 GPU 장비가 있어야합니다.
2) 분류 카테고리 파일은 .xlsx이어야 합니다.
3) 시트명은 'context'로 대소문자까지 일치해야합니다.
4) 첫번째 컬럼에 카테고리, 두번째 컬럼에 카테고리 설명이 위치해야합니다.

## 활용 방법
1) .env 파일을 사용자의 환경에 맞게 작성
2) .xlsx 카테고리 파일 준비
3) python3 -m venv venv
4) source venv/bin/activate
5) pip install -r requirements.txt
6) uvicorn api.main:api --host 0.0.0.0 --port 8000 --reload
7) api 호출 도구를 활용하여 /classification/text 경로로 호출

## 유의 사항
현재 개발 진행 중인 프로젝트로 파일 업로드는 아직 구현 전입니다.
JSON 형식으로 message에 텍스트를 작성 후 API 호출하면 해당 텍스트에 지정한 분류 카테고리 파일에서 중 가장 적합한 카테고리를 찾아 응답합니다.
