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


def get_vector_store():
    """ChromaDB 벡터 저장소 로드."""
    if not os.path.exists(CHROMA_DATA_PATH):
        raise FileNotFoundError(
            f"ChromaDB path not found: {CHROMA_DATA_PATH}. "
            f"Please run the embedding script first (src/embedding/embedder.py)."
        )

    embedding_function = get_embedding_function()
    vector_store = Chroma(
        persist_directory=CHROMA_DATA_PATH,
        embedding_function=embedding_function,
        collection_name=COLLECTION_NAME
    )
    return vector_store

from langchain_community.llms import Ollama

def get_llm():
    """사용할 LLM 설정 및 반환."""
    return Ollama(model="gemma:2b", base_url="http://ollama:11434")


class CareerRAG:
    def __init__(self):
        print("Initializing CareerRAG...")
        self.vector_store = get_vector_store()
        self.llm = get_llm()
        self.qa_chain = self._create_qa_chain()
        print("CareerRAG initialized successfully.")

    def _create_qa_chain(self):
        """RAG 체인 생성."""
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        
        # 프롬프트 템플릿 정의 (한국어)
        prompt_template = '''
        당신은 채용 공고를 요약해주는 유용한 어시스턴트입니다.
        오직 다음에 주어지는 채용 공고 정보만을 바탕으로 사용자의 질문에 답변하세요.
        찾은 공고들을 요약하고, 각 공고의 핵심 내용을 정리해서 보여주세요.
        만약 컨텍스트에서 관련 공고를 찾을 수 없다면, "관련 채용 정보를 찾을 수 없습니다."라고 답변하세요.

        {context}

        질문: {question}
        답변:
        '''
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # RetrievalQA 체인 생성
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        return qa_chain

    def ask(self, query: str):
        """질문 처리 및 답변, 출처 반환."""
        print(f"\n[Query] {query}")
        result = self.qa_chain.invoke({"query": query})
        
        print("\n[Answer]")
        print(result["result"])
        
        print("\n[Source Documents]")
        for doc in result["source_documents"]:
            print(f"  - {doc.metadata.get('cleaned_title', 'N/A')} ({doc.metadata.get('company', 'N/A')})")
            print(f"    Link: {doc.metadata.get('link', 'N/A')}")
        
        return result