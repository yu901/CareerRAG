# 1. 베이스 이미지 설정
FROM python:3.12-slim

# 2. 시스템 환경 변수 설정
ENV PYTHONUNBUFFERED=1

# 3. Airflow 환경 변수 설정
ENV AIRFLOW_HOME=/opt/airflow
ENV AIRFLOW__CORE__DAGS_FOLDER=/opt/airflow/dags
ENV AIRFLOW__CORE__LOAD_EXAMPLES=false
ENV AIRFLOW__CORE__EXECUTOR=LocalExecutor

# 4. 작업 디렉토리 설정
WORKDIR /opt/airflow

# 5. 시스템 패키지 업데이트 및 의존성 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libpq-dev \
    default-jdk-headless \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 5-1. JAVA_HOME 환경 변수 설정
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH=$PATH:$JAVA_HOME/bin

# 6. requirements.txt 복사 및 Python 의존성 설치
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 7. 소스 코드 및 DAGs 복사
COPY src/ /opt/airflow/src/
COPY dags/ /opt/airflow/dags/

# 7-1. Airflow 고정 비밀번호 설정
RUN mkdir -p /opt/airflow && \
    echo '{"admin": "admin"}' > /opt/airflow/simple_auth_manager_passwords.json.generated

# 8. 포트 노출
EXPOSE 8000 8080 5555
