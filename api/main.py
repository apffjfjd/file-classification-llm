import torch
from fastapi import FastAPI
from dotenv import load_dotenv
from api.routers import classification

if torch.cuda.is_available():
    print(f"CUDA is available. GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU available or CUDA not installed.")

api = FastAPI()

# 엔드포인트 라우터 등록
api.include_router(classification.router, prefix="/classification")

# 서버 실행 시 기본 라우트 확인용 엔드포인트
@api.get("/")
async def root():
    return {"message": "AI 기반 텍스트 분류 시작"}
