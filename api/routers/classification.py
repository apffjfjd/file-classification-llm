import logging
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from api.core.models.model_factory import ModelFactory
from api.core.preprocessor import Preprocessor
from api.core.vectorizer import Vectorizer
from api.core.util import Util

router = APIRouter()

# 로거 설정
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChatRequest(BaseModel):
    message: str
    user_id: str
    
def setup_classification_prompt(message: str, categories: list) -> str:
        formatted_categories = "\n".join(categories)
        prompt_template = f"""
        주어진 입력 텍스트에 가장 적합한 카테고리를 아래 카테고리 리스트에서만 **정확히 하나** 선택하세요. **리스트에 없는 새로운 답변을 절대 생성하지 마세요.** 

        입력 텍스트: "{message}"

        카테고리 리스트:
        [{formatted_categories}]

        **주의사항:**
        - 반드시 위의 카테고리 리스트에서만 하나를 선택해야 합니다.
        - 다른 문장을 생성하지 말고, **카테고리명만 한 줄로 답변**하세요.
        - 만약 리스트에 적합한 카테고리가 없다면, "적합한 카테고리 없음"이라고 답변하세요.
        """
        return prompt_template

@router.post("/text")
def classify_text(request: ChatRequest):

    Util.load_env()
    
    model_name = Util.get_model_name()
    model_path = Util.get_model_path()
    hf_api_token = Util.get_hf_api_token()
    context_path = Util.get_context_path()

    # api_model = ModelFactory.create_model_handler(model_name, model_path, hf_api_token)
    model_handler = ModelFactory.create_model_handler(model_name, model_path, hf_api_token)
    model = model_handler.get_model()
    
    logger.debug(f"불러온 파일 경로: {context_path} (타입: {type(context_path)})")
    
    context = Preprocessor(context_path)

    params = {
        'context_path': context
    }
    vectorizer = Vectorizer(**params)
    # 인스턴스를 통해 메서드 호출
    top_categories = vectorizer.get_top_k_similar_categories(request.message, k=4)
    # 정규 표현식을 이용해 불필요한 부분 제거 및 카테고리 추출
    
    # 프롬프트 생성
    prompt_template = setup_classification_prompt(request.message, top_categories)

    # 토크나이저 초기화
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(prompt_template, return_tensors="pt")

    # 모델의 디바이스로 입력 이동
    device = next(model.parameters()).device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # 모델 출력 생성
    outputs = model.generate(**inputs, max_new_tokens=50)

    # 출력을 텍스트로 디코딩
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"response": response_text}




