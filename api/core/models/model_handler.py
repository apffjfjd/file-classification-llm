import torch
import os
import logging
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv
from api.core.preprocessor import Preprocessor
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

# 로거 설정
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelHandler:
    def __init__(self, model_name: str, model_path: str, hf_api_token: str):
        self.model_name = model_name
        self.model_path = model_path
        self.hf_api_token = hf_api_token
        self.model = None
        self.tokenizer = None

    def is_model_downloaded(self) -> bool:
        """모델이 로컬에 존재하는지 확인합니다."""
        if not os.path.isdir(self.model_path):
            return False
        
        required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
        
        for file_name in required_files:
            file_path = os.path.join(self.model_path, file_name)
            if not os.path.exists(file_path):
                return False
        
        model_files_exist = any(
            file_name.endswith(('.bin', '.safetensors', '.pt'))
            for file_name in os.listdir(self.model_path)
            if 'model' in file_name or 'pytorch_model' in file_name
        )
        
        return model_files_exist

    def load_model(self):
        """모델을 로드합니다. 다운로드가 필요하다면 이를 알립니다."""
        if not self.is_model_downloaded():
            logger.info("모델이 로컬에 존재하지 않습니다. 다운로드가 필요합니다.")
            return None

        logger.info(f"{self.model_path}에서 모델을 로드합니다.")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto"
        )

    def download_model(self, quantize: bool = True):
        """모델을 다운로드하고 필요에 따라 양자화합니다."""
        if quantize:
            self._quantize_model()
        else:
            self._download_without_quantization()

    def _quantize_model(self):
        """모델을 다운로드하고 양자화합니다."""
        logger.info(f"{self.model_name} 모델을 양자화하여 다운로드 중입니다...")

        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_auth_token=self.hf_api_token
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            use_auth_token=self.hf_api_token
        )

        logger.info(f"모델이 양자화되었습니다. 로컬에 저장을 진행합니다.")
        self.save_model_to_local()

    def _download_without_quantization(self):
        """양자화 없이 모델을 다운로드합니다."""
        logger.info(f"{self.model_name} 모델을 다운로드 중입니다...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_auth_token=self.hf_api_token
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            use_auth_token=self.hf_api_token
        )

        logger.info(f"모델이 다운로드되었습니다. 로컬에 저장을 진행합니다.")
        self.save_model_to_local()

    def save_model_to_local(self):
        """모델과 토크나이저를 로컬에 저장합니다."""
        self.tokenizer.save_pretrained(self.model_path)
        self.model.save_pretrained(self.model_path)
        logger.info(f"모델과 토크나이저가 {self.model_path}에 저장되었습니다.")

    def get_model(self):
        if self.model is None:
            self.load_model()
        return self.model

