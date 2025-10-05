# 1. 베이스 이미지 설정
FROM python:3.12-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 시스템 패키지 업데이트 및 git 설치
RUN apt-get update && apt-get install -y git

# 4. requirements.txt 복사 및 의존성 설치
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 5. 소스 코드 복사
COPY src/ src/

# 6. 데이터 및 모델/DB 경로를 위한 볼륨 마운트 지점 설정
# RUN mkdir -p /app/data /app/chroma

# 7. 포트 노출
EXPOSE 8000

# 8. 애플리케이션 실행
CMD ["uvicorn", "src.main.python.main:app", "--host", "0.0.0.0", "--port", "8000"]
