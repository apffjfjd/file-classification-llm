import faiss
# import hashlib
import logging
import os
import pickle

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from api.core.util import Util

# 로거 설정
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Vectorizer:
    def __init__(self, **kwargs):
        self.faiss_db_path = kwargs.get('faiss_db_path', Util.get_faiss_db_path())
        self.embedding_model = kwargs.get('embedding_model', HuggingFaceEmbeddings())
        self.context_path = kwargs.get('context_path')
        self.faiss_index = kwargs.get('faiss_index', None)
        self.faiss_file_path = kwargs.get('faiss_file_path', os.path.join(Util.get_faiss_db_path(), 'faiss_index.pkl'))

    # 1. 엑셀에서 카테고리 로드 및 텍스트 벡터화
    def load_and_save_vectors_from_context(self, context):
        # context 객체의 메서드를 사용하여 데이터를 가져옴
        categories = context.get_categories()
        descriptions = context.get_descriptions()
        
        # 텍스트를 벡터화하고 FAISS 인덱스 생성
        self.faiss_index = FAISS.from_texts(descriptions, self.embedding_model, metadatas=[{"카테고리": cat} for cat in categories])

        # FAISS 인덱스를 파일로 저장
        self._save_faiss_index()
        
        return self.faiss_index

    def _save_faiss_index(self):
        if not self.faiss_index:
            raise ValueError("FAISS 인덱스가 초기화되지 않았습니다. 저장할 인덱스가 없습니다.")
        
        # 디렉토리가 존재하지 않으면 생성
        os.makedirs(self.faiss_db_path, exist_ok=True)

        with open(self.faiss_file_path, 'wb') as f:
            pickle.dump(self.faiss_index, f)
        print(f"FAISS 인덱스가 {self.faiss_file_path}에 저장되었습니다.")

    # 2. 쿼리 벡터화 및 상위 4개 유사 카테고리 검색
    def get_top_k_similar_categories(self, query, k=4):
        # FAISS 인덱스가 로드되었는지 확인
        if self.faiss_index is None:
            # 인덱스를 파일에서 로드
            with open(self.faiss_db_path, 'rb') as f:
                self.faiss_index = pickle.load(f)
            logger.info(f"FAISS 인덱스가 {self.faiss_db_path}에서 로드되었습니다.")
        
        # 입력 텍스트를 벡터화
        document_vector = self.embedding_model.embed_documents([query])[0]
        similar_docs = self.faiss_index.similarity_search_by_vector(document_vector, k=k)
        
        # 유사 카테고리 리스트 반환
        top_k_categories = [doc.metadata["카테고리"] for doc in similar_docs]
        return top_k_categories