# CareerRAG

**CareerRAG**는 채용공고 데이터를 수집하고, Vector Database에 임베딩하여 저장한 뒤,  
RAG(Retrieval-Augmented Generation) 기반 LLM을 활용해 맞춤형 채용 검색과 커리어 인사이트를 제공하는 미니 프로젝트입니다.

---

## 목표
- 최신 채용공고 데이터를 벡터화하여 **효율적인 검색** 지원
- **RAG 기반 LLM**으로 자연어 질의 응답 제공
- **컨테이너, Spark, Vector DB** 등을 적용

---

## 기술 스택
- **Python 3.12**
- **Vector Database**: ChromaDB
- **LLM 프레임워크**: LangChain
- **컨테이너**: Docker
- **PySpark**: 데이터 전처리 및 텍스트 처리

---

## 데이터 수집 방법

채용공고 데이터는 [사람인](https://www.saramin.co.kr) 사이트에서 수집합니다.  
URL은 다음과 같은 형식입니다:

https://www.saramin.co.kr/zf_user/search/recruit?searchword=데이터엔지니어&recruitPage=2


- `searchword`: 검색 키워드 (예: 데이터엔지니어)  
- `recruitPage`: 페이지 번호 (1~100)  

프로젝트에서는 1페이지부터 100페이지까지 순회하며 각 페이지의 채용 공고 정보를 수집하고, 이를 벡터화하여 Vector DB에 저장합니다.  

## 데이터 수집 방법

채용공고 데이터는 [사람인](https://www.saramin.co.kr) 사이트에서 수집합니다.  
URL은 다음과 같은 형식입니다:

https://www.saramin.co.kr/zf_user/search/recruit?searchword=데이터엔지니어&recruitPage=2


- `searchword`: 검색 키워드 (예: 데이터엔지니어)  
- `recruitPage`: 페이지 번호 (1~100)  

프로젝트에서는 1페이지부터 100페이지까지 순회하며 각 페이지의 채용 공고 정보를 수집하고, 이를 벡터화하여 Vector DB에 저장합니다.  
---

## 프로젝트 실행 방법

이 프로젝트는 Docker를 사용하거나, 로컬 환경에서 직접 Python 스크립트를 실행하는 두 가지 방법으로 실행할 수 있습니다.

### 1. Docker를 사용하여 실행

프로젝트에 필요한 모든 환경이 Docker 이미지에 포함되어 있어 가장 간편하게 실행할 수 있습니다.

**요구사항:**
- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)

**실행 단계:**

1.  프로젝트의 루트 디렉토리에서 다음 명령어를 실행하여 Docker 이미지를 빌드하고 컨테이너를 시작합니다.

    ```bash
    docker compose up --build
    ```

2.  위 명령어를 실행하면 다음 과정이 순서대로 자동으로 진행됩니다.
    - `Dockerfile`을 기반으로 이미지를 빌드합니다.
    - 채용 공고 데이터 수집 (`scraper.py`)
    - 데이터 전처리 (`parser.py`)
    - 데이터 임베딩 및 Vector DB 저장 (`embedder.py`)
    - RAG 기반 질의응답(QA) 시스템 실행 (`qa.py`)

### Docker 환경에서 QA 시스템 테스트

`docker compose up --build` 명령으로 컨테이너가 실행된 후에는, 다음 두 가지 방법으로 QA 시스템을 테스트할 수 있습니다.

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

### 2. 로컬 환경에서 직접 실행

로컬에 Python 개발 환경이 구성되어 있는 경우, 직접 스크립트를 실행할 수 있습니다.

**요구사항:**
- Python 3.12
- `pip`

**실행 단계:**

1.  Python 가상 환경을 생성하고 활성화합니다.

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  `requirements.txt` 파일을 사용하여 필요한 라이브러리를 설치합니다.

    ```bash
    pip install -r requirements.txt
    ```

3.  `main.py`를 실행하여 전체 데이터 파이프라인 및 QA 시스템을 시작합니다.

    ```bash
    python src/main/python/main.py
    ```