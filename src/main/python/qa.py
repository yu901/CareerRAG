# sqlite3/chromadb fix
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate
from langchain_core.embeddings import Embeddings
import os
from src.main.python.utils import load_config
from typing import List

# 설정 로드
config = load_config()
CHROMA_DATA_PATH = config['chroma_path']
COLLECTION_NAME = config['collection_name']
EMBEDDING_MODEL = config['embedding_model']


class ChromaEmbeddingWrapper(Embeddings):
    """ChromaDB의 SentenceTransformerEmbeddingFunction을 LangChain과 호환되도록 래핑"""

    def __init__(self, model_name: str):
        self.embedding_function = SentenceTransformerEmbeddingFunction(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서 리스트를 임베딩"""
        return self.embedding_function(texts)

    def embed_query(self, text: str) -> List[float]:
        """단일 쿼리를 임베딩"""
        return self.embedding_function([text])[0]


def get_embedding_function():
    """임베딩 함수 로드 - embedder.py와 동일한 방식 사용."""
    return ChromaEmbeddingWrapper(model_name=EMBEDDING_MODEL)


def get_chroma_collection():
    """ChromaDB 컬렉션 직접 로드."""
    if not os.path.exists(CHROMA_DATA_PATH):
        raise FileNotFoundError(
            f"ChromaDB path not found: {CHROMA_DATA_PATH}. "
            f"Please run the embedding script first (src/embedding/embedder.py)."
        )

    # ChromaDB 클라이언트 직접 사용
    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    # 컬렉션 로드
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )
    return collection

from langchain_community.llms import Ollama

def get_llm():
    """사용할 LLM 설정 및 반환."""
    return Ollama(model="gemma:2b", base_url="http://ollama:11434")


class CareerRAG:
    def __init__(self):
        print("Initializing CareerRAG...")
        self.collection = get_chroma_collection()
        self.llm = get_llm()
        print("CareerRAG initialized successfully.")

    def ask(self, query: str):
        """질문 처리 및 답변, 출처 반환."""
        print(f"\n[Query] {query}")

        # 질문 유형 판단 및 검색 개수 조정
        n_results = 5

        # "필수 역량", "자격요건", "역량" 등의 키워드가 있으면 더 많은 공고 검색
        analysis_keywords = ["필수", "역량", "자격", "요건", "스킬", "기술"]
        if any(keyword in query for keyword in analysis_keywords):
            n_results = 15  # 더 많은 공고로 패턴 분석

        # "내가", "나는", "경험" 등의 키워드가 있으면 매칭용
        matching_keywords = ["내가", "나는", "저는", "경험", "할 수 있", "가능"]
        if any(keyword in query for keyword in matching_keywords):
            n_results = 10  # 매칭을 위해 적당히 많이

        # ChromaDB에서 직접 검색
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

        if not results['documents'] or not results['documents'][0]:
            return {
                "result": "관련 채용 정보를 찾을 수 없습니다.",
                "source_documents": []
            }

        # 컨텍스트 생성
        context_parts = []
        source_docs = []

        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            context_parts.append(f"[공고 {i+1}]")
            context_parts.append(f"회사: {metadata.get('company', 'N/A')}")
            context_parts.append(f"제목: {metadata.get('cleaned_title', 'N/A')}")
            context_parts.append(f"담당업무: {metadata.get('responsibilities', 'N/A')}")
            context_parts.append(f"자격요건: {metadata.get('qualifications', 'N/A')}")
            context_parts.append(f"우대사항: {metadata.get('preferences', 'N/A')}")
            context_parts.append("")

            source_docs.append(metadata)

        context = "\n".join(context_parts)

        # LLM 프롬프트
        prompt = f"""당신은 채용 공고를 분석하고 커리어 조언을 제공하는 전문 어시스턴트입니다.
다음에 주어지는 채용 공고 정보를 바탕으로 사용자의 질문에 답변하세요.

질문 유형에 따라 다음과 같이 답변하세요:

1. **특정 공고 검색** (예: "서울 데이터 엔지니어")
   - 관련 공고들을 나열하고 각 공고의 핵심 내용 요약

2. **필수 역량/자격요건 분석** (예: "데이터 엔지니어 필수 역량은?")
   - 모든 공고의 자격요건과 우대사항을 분석
   - 공통적으로 등장하는 역량을 빈도수와 함께 표시
   - 형식: "역량명 (N개 공고 중 M개에서 요구)"
   - 빈도가 높은 순서대로 정리
   - 필수 요구사항과 우대사항을 구분하여 표시

3. **개인 역량 매칭** (예: "Kafka, Spark 경험이 있는데 적합한 공고는?")
   - 사용자가 언급한 기술/역량이 포함된 공고만 선별
   - 매칭도(%)를 함께 표시
   - 각 공고별로 매칭되는 구체적인 요구사항 명시
   - 필수 요구사항 매칭 > 우대사항 매칭 순으로 우선순위 부여

채용 공고 정보 (총 {len(source_docs)}개):
{context}

질문: {query}

답변 형식을 반드시 준수하여 구조화된 답변을 제공하세요:"""

        # LLM 답변 생성
        answer = self.llm.invoke(prompt)

        print("\n[Answer]")
        print(answer)

        print("\n[Source Documents]")
        for doc in source_docs:
            print(f"  - {doc.get('cleaned_title', 'N/A')} ({doc.get('company', 'N/A')})")
            print(f"    Link: {doc.get('link', 'N/A')}")

        return {
            "result": answer,
            "source_documents": source_docs
        }