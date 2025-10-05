# sqlite3/chromadb fix
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate
import os

# 임베딩 및 ChromaDB 설정
CHROMA_DATA_PATH = "chroma/career_rag_db"
COLLECTION_NAME = "job_postings"

def get_embedding_function():
    """임베딩 함수 로드."""
    return HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

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

if __name__ == '__main__':
    # 시스템 초기화
    try:
        rag_system = CareerRAG()
        
        # 테스트 질문
        rag_system.ask("서울에서 근무하는 데이터 엔지니어 채용 정보 알려줘")
        rag_system.ask("머신러닝 엔지니어 직무는 어떤 일을 해?") # DB에 없는 내용

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")