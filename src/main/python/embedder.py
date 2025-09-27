# ChromaDB sqlite3 버전 호환성 우회
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import findspark
findspark.init()
from pyspark.sql import SparkSession
import os

# ChromaDB 클라이언트 설정 (로컬 지속성 사용)
CHROMA_DATA_PATH = "chroma/career_rag_db"
COLLECTION_NAME = "job_postings"

def get_embedding_function():
    """ChromaDB 임베딩 모델 로드."""
    # 한국어 임베딩 모델 설정 (최초 실행 시 다운로드)
    embedding_function = SentenceTransformerEmbeddingFunction(model_name="jhgan/ko-sroberta-multitask")
    return embedding_function

def main():
    """데이터 임베딩 및 ChromaDB 저장."""
    # Spark 세션 초기화
    spark = SparkSession.builder \
        .appName("CareerRAG Embedding") \
        .master("local[*]") \
        .getOrCreate()

    # 처리된 데이터 로드
    processed_data_path = "data/processed_postings.parquet"
    try:
        df = spark.read.parquet(processed_data_path)
    except Exception as e:
        print(f"Error reading data from {processed_data_path}: {e}")
        spark.stop()
        return

    # Pandas DataFrame 변환 (대규모 데이터는 Spark 직접 처리 고려)
    pandas_df = df.toPandas()
    spark.stop()

    if pandas_df.empty:
        print("No data to embed. Exiting.")
        return

    # 임베딩 함수 로드
    embedding_function = get_embedding_function()

    # ChromaDB 클라이언트 초기화
    if not os.path.exists(CHROMA_DATA_PATH):
        os.makedirs(CHROMA_DATA_PATH)
    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

    # 컬렉션 생성 또는 조회
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function, # LangChain 임베딩 함수 직접 사용 불가
        metadata={"hnsw:space": "cosine"}  # 코사인 유사도 설정
    )

    # 임베딩 텍스트 준비 (제목과 키워드 결합)
    documents = (pandas_df["cleaned_title"] + " (" + pandas_df["keyword"] + ")").tolist()
    metadatas = pandas_df.to_dict('records')
    ids = [f"post_{i}" for i in range(len(documents))]

    print(f"Start embedding {len(documents)} documents...")

    # ChromaDB에 데이터 추가 (upsert, 대규모 데이터는 배치 처리 권장)
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    print(f"Successfully embedded and stored {len(documents)} documents in ChromaDB collection '{COLLECTION_NAME}'.")
    print(f"DB Path: {CHROMA_DATA_PATH}")

if __name__ == "__main__":
    main()
