# CareerRAG

**CareerRAG**는 채용공고 데이터를 수집하고, Vector Database에 임베딩하여 저장한 뒤,  
RAG(Retrieval-Augmented Generation) 기반 LLM을 활용해 맞춤형 채용 검색과 커리어 인사이트를 제공하는 미니 프로젝트입니다.

이 프로젝트는 데이터 파이프라인 관리를 위해 **Apache Airflow**를 사용합니다.

---

## 아키텍처

- **Data Pipeline (Airflow)**: `scrape` -> `parse` -> `embed` 순서로 실행되는 데이터 파이프라인을 DAG로 관리합니다.
- **QA System (FastAPI)**: RAG 기반의 질의응답 API를 제공합니다.
- **LLM**: `ollama/gemma:2b` 모델을 사용합니다.
- **Vector DB**: `ChromaDB`를 사용하여 임베딩된 데이터를 저장합니다.
- **Orchestration**: `Docker Compose`를 사용하여 모든 서비스(Airflow, FastAPI, Ollama, Postgres)를 실행합니다.

---

## 데이터 파이프라인

Apache Airflow를 사용하여 채용 공고 데이터의 수집, 전처리, 임베딩 과정을 관리합니다. 파이프라인은 `career_rag_pipeline`이라는 DAG로 정의되어 있으며, 다음과 같은 3개의 태스크로 구성됩니다.

1.  **`scrape_task`**
    - **스크립트**: `src/main/python/scraper.py`
    - **기능**: 사람인(Saramin) 사이트에서 채용 공고를 스크래핑하여 원본 데이터를 수집합니다.

2.  **`parse_task`**
    - **스크립트**: `src/main/python/parser.py`
    - **기능**: 수집된 데이터를 PySpark를 사용하여 정제하고, 분석에 용이한 형태로 가공합니다.

3.  **`embed_task`**
    - **스크립트**: `src/main/python/embedder.py`
    - **기능**: 정제된 데이터를 `gemma:2b` 모델을 통해 텍스트 임베딩으로 변환하고, ChromaDB에 저장합니다.

---

## 기술 스택
- **Python 3.12**
- **Orchestrator**: Apache Airflow
- **Vector Database**: ChromaDB
- **LLM 프레임워크**: LangChain
- **컨테이너**: Docker, Docker Compose
- **PySpark**: 데이터 전처리 및 텍스트 처리

---

## 프로젝트 실행 방법

이 프로젝트는 Docker Compose를 사용하여 모든 서비스를 한번에 실행합니다.

**요구사항:**
- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)

### 1. 서비스 시작

프로젝트의 루트 디렉토리에서 다음 명령어를 실행하여 모든 서비스를 시작합니다. 이전에 사용되지 않는 컨테이너를 정리하기 위해 `--remove-orphans` 플래그를 함께 사용하는 것을 권장합니다.

```bash
docker compose up --build --remove-orphans
```

이 명령은 다음 서비스들을 실행합니다:
- **Postgres**: Airflow 메타데이터 저장소
- **Airflow Init**: Airflow DB 초기화
- **Airflow Standalone**: Airflow UI, 스케줄러, 트리거 등을 포함하는 올인원 서비스 (http://localhost:8080)
- **FastAPI App**: QA API 서버 (http://localhost:8000)
- **Ollama**: LLM 모델 서버


### 2. 데이터 파이프라인 실행 (DAG 트리거)

서비스가 모두 시작되면, Airflow UI에서 데이터 파이프라인을 직접 실행해야 합니다.

1.  **Airflow UI 접속**: 웹 브라우저에서 **http://localhost:8080**으로 접속하여 Airflow UI를 엽니다.

2.  **로그인**: `admin` / `admin` (Dockerfile에서 고정 비밀번호 설정)

3.  **DAG 실행**: DAG 목록에서 `career_rag_pipeline`을 찾습니다. 해당 DAG의 오른쪽에 있는 **실행(▶) 버튼**을 눌러 DAG를 수동으로 트리거합니다.

4.  **실행 확인**: DAG 이름을 클릭하여 Grid View 또는 Graph View에서 파이프라인의 실행 상태를 확인할 수 있습니다. 모든 단계(scrape, parse, embed)가 초록색으로 바뀌면 성공적으로 완료된 것입니다.


### 3. QA 시스템 테스트

데이터 파이프라인이 성공적으로 완료된 후, QA 시스템을 테스트할 수 있습니다.

#### 1. `curl`을 사용하여 터미널에서 테스트

다음 `curl` 명령어를 터미널에 실행하여 QA 시스템에 질문을 보낼 수 있습니다.

```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"query": "서울에서 근무하는 데이터 엔지니어 채용 정보 알려줘"}'
```

#### 2. 웹 브라우저에서 API 문서로 테스트

웹 브라우저에서 `http://localhost:8000/docs`로 접속하면, FastAPI가 제공하는 대화형 API 문서 페이지가 나타납니다.

- `/ask` 엔드포인트를 찾아서 확장합니다.
- "Try it out" 버튼을 클릭합니다.
- "Request body"에 질문을 JSON 형식으로 입력하고 "Execute" 버튼을 누르면 결과를 확인할 수 있습니다.