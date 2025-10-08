# ChromaDB sqlite3 버전 호환성 우회
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import findspark
findspark.init()
from pyspark.sql import SparkSession
from src.main.python.utils import load_config, ensure_dir


def main(input_dir=None, chroma_path=None, collection_name=None, embedding_model=None, **kwargs):
    """데이터 임베딩 및 ChromaDB 저장."""
    # Airflow에서 전달된 파라미터가 있으면 사용, 없으면 설정 파일에서 로드
    if any(param is None for param in [input_dir, chroma_path, collection_name, embedding_model]):
        config = load_config()
        input_dir = input_dir or config['embedder_input_dir']
        chroma_path = chroma_path or config['chroma_path']
        collection_name = collection_name or config['collection_name']
        model_name = embedding_model or config['embedding_model']
    else:
        model_name = embedding_model
    
    # Spark 세션 초기화
    spark = SparkSession.builder \
        .appName("CareerRAG Embedding") \
        .master("local[*]") \
        .getOrCreate()

    try:
        # 처리된 데이터 로드
        df = spark.read.parquet(f"{input_dir}/processed_postings.parquet")
        pandas_df = df.toPandas()

        if pandas_df.empty:
            print("No data to embed")
            return

        print(f"Embedding {len(pandas_df)} records")

        # 임베딩 함수 로드
        embedding_function = SentenceTransformerEmbeddingFunction(model_name=model_name)

        # ChromaDB 클라이언트 초기화
        ensure_dir(chroma_path)
        client = chromadb.PersistentClient(path=chroma_path)

        # 컬렉션 생성 (기존 컬렉션이 있으면 삭제 후 재생성)
        try:
            existing_collection = client.get_collection(name=collection_name)
            # 기존 컬렉션의 임베딩 차원 확인
            existing_dim = existing_collection._embedding_function
            # 차원이 다르면 삭제하고 재생성
            print(f"Deleting existing collection '{collection_name}' to recreate with new embedding dimension")
            client.delete_collection(name=collection_name)
        except:
            pass

        collection = client.create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"}
        )

        # 임베딩 데이터 준비 - 모든 중요 필드 포함
        documents = (
            pandas_df["cleaned_title"] + " | " +
            pandas_df["company"] + " | " +
            pandas_df["responsibilities"].fillna("") + " | " +
            pandas_df["qualifications"].fillna("") + " | " +
            pandas_df["preferences"].fillna("") + " | " +
            pandas_df["keyword"]
        ).tolist()
        metadatas = pandas_df.to_dict('records')
        ids = [f"post_{i}" for i in range(len(documents))]

        # 배치 단위로 ChromaDB에 데이터 추가
        batch_size = 500
        for i in range(0, len(documents), batch_size):
            batch_end = min(i + batch_size, len(documents))
            collection.upsert(
                ids=ids[i:batch_end],
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )

        print(f"Successfully embedded {len(documents)} documents")
        print(f"DB Path: {chroma_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        spark.stop()

if __name__ == "__main__":
    main()