from __future__ import annotations

import pendulum
from datetime import timedelta

from airflow.models.dag import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.models import Variable

import sys
import os

# 프로젝트 루트를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main.python import scraper, parser, embedder
from src.main.python.utils import load_config

# 기본 설정값
default_args = {
    'owner': 'career-rag',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id="career_rag_pipeline",
    default_args=default_args,
    schedule=None,  # 수동 트리거
    start_date=pendulum.datetime(2023, 1, 1, tz="Asia/Seoul"),
    catchup=False,
    tags=["rag", "pipeline"],
    doc_md="""
    ### CareerRAG 데이터 파이프라인

    이 DAG는 Airflow Variables를 통해 설정을 관리하며 다음 작업을 수행합니다:
    
    **Airflow Variables 설정:**
    - `search_keywords`: 검색할 키워드 목록 (JSON 배열)
    - `pages_per_keyword`: 키워드당 스크래핑 페이지 수
    - `scraper_output_dir`: 스크래핑 결과 저장 디렉토리
    - `use_summarization`: 텍스트 요약 사용 여부 (boolean)
    - `chroma_path`: ChromaDB 저장 경로
    - `embedding_model`: 사용할 임베딩 모델명
    
    **태스크:**
    - **scrape_task**: 채용 공고 스크래핑
    - **parse_task**: 데이터 전처리 및 정제
    - **embed_task**: 벡터 임베딩 및 ChromaDB 저장
    """,
) as dag:

    # 설정 로드
    config = load_config()

    scrape_task = PythonOperator(
        task_id="scrape_task",
        python_callable=scraper.main,
        op_kwargs={
            'keywords': config.get('search_keywords'),
            'pages_per_keyword': config.get('pages_per_keyword'),
            'output_dir': config.get('scraper_output_dir')
        }
    )

    parse_task = PythonOperator(
        task_id="parse_task",
        python_callable=parser.main,
        op_kwargs={
            'input_dir': config.get('parser_input_dir'),
            'output_dir': config.get('parser_output_dir'),
            'use_summarization': config.get('use_summarization')
        }
    )

    embed_task = PythonOperator(
        task_id="embed_task",
        python_callable=embedder.main,
        op_kwargs={
            'input_dir': config.get('embedder_input_dir'),
            'chroma_path': config.get('chroma_path'),
            'collection_name': config.get('collection_name'),
            'embedding_model': config.get('embedding_model')
        }
    )

    scrape_task >> parse_task >> embed_task
