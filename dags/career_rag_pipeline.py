from __future__ import annotations

import pendulum

from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator

# 파이프라인 스크립트 임포트
# Airflow는 DAG 파일이 있는 디렉토리를 sys.path에 추가하므로, 
# 프로젝트 구조에 맞게 경로를 조정해야 할 수 있습니다.
# 이 예제에서는 src/main/python이 PYTHONPATH에 있다고 가정합니다.
# 만약 그렇지 않다면, sys.path를 직접 조작해야 합니다.
import sys
import os

# 프로젝트 루트를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main.python import scraper, parser, embedder

with DAG(
    dag_id="career_rag_pipeline",
    schedule=None,  # 수동 트리거
    start_date=pendulum.datetime(2023, 1, 1, tz="Asia/Seoul"),
    catchup=False,
    tags=["rag", "pipeline"],
    doc_md="""
    ### CareerRAG 데이터 파이프라인

    이 DAG는 채용 공고 데이터를 수집, 전처리, 임베딩하는 전체 파이프라인을 관리합니다.
    - **scrape_task**: `scraper.py`를 실행하여 채용 공고를 스크래핑합니다.
    - **parse_task**: `parser.py`를 실행하여 수집된 데이터를 정제하고 처리합니다.
    - **embed_task**: `embedder.py`를 실행하여 처리된 데이터를 벡터화하고 ChromaDB에 저장합니다.
    """,
) as dag:
    scrape_task = PythonOperator(
        task_id="scrape_task",
        python_callable=scraper.main,
    )

    parse_task = PythonOperator(
        task_id="parse_task",
        python_callable=parser.main,
    )

    embed_task = PythonOperator(
        task_id="embed_task",
        python_callable=embedder.main,
    )

    scrape_task >> parse_task >> embed_task
