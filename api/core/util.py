import os
from dotenv import load_dotenv

class Util:
    # 정적 메서드를 사용하여 환경 변수 값을 반환
    @staticmethod
    def load_env():
        load_dotenv()  # .env 파일을 로드

    @staticmethod
    def get_hf_api_token():
        return os.getenv("HF_API_TOKEN")

    @staticmethod
    def get_model_name():
        return os.getenv("MODEL_NAME")

    @staticmethod
    def get_model_path():
        return os.getenv("MODEL_PATH")

    @staticmethod
    def get_context_path():
        return os.getenv("CONTEXT_PATH")

    @staticmethod
    def get_faiss_db_path():
        return os.getenv("FAISS_DB_PATH")
