"""
CareerRAG 파이프라인 공통 유틸리티 함수들
"""
import json
from pathlib import Path
from typing import Dict, Any


def load_config(file_path="config/default_variables.json") -> Dict[str, Any]:
    """설정을 로드합니다. Airflow 환경에서는 Variables 우선, 그 외에는 파일 기반"""
    # 기본값 파일 로드
    config_path = Path(file_path)
    defaults = {}
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            defaults = json.load(f)

    # Airflow Variable 로드 시도
    try:
        from airflow.models import Variable
        import logging

        # Variable 조회 시 발생하는 에러 로그 억제
        logging.getLogger('airflow.models.variable').setLevel(logging.CRITICAL)
        logging.getLogger('airflow.sdk.api.client').setLevel(logging.CRITICAL)

        config = {}
        for key, default_value in defaults.items():
            config[key] = Variable.get(key, default_var=default_value)

        return config
    except Exception:
        return defaults


def ensure_dir(directory: str) -> None:
    """디렉토리가 존재하지 않으면 생성합니다."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    """텍스트 정제 (불필요한 문자, 공백, HTML 태그 제거)."""
    import re
    
    if not text:
        return ""
    
    text = re.sub(r'<[^>]+>', '', text)  # HTML 태그 제거
    text = re.sub(r'\s+', ' ', text)     # 중복 공백 제거
    text = text.strip()                  # 앞뒤 공백 제거
    
    return text