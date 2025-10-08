import findspark
findspark.init()
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, StringType
import re
from src.main.python.utils import load_config, ensure_dir, clean_text


def process_partition(iterator, use_summarization=True):
    # 파티션당 한 번만 LLM 로딩
    extraction_llm = None
    if use_summarization:
        try:
            from langchain_community.llms import Ollama
            from langchain_core.prompts import PromptTemplate
            import logging

            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)

            # Ollama LLM 초기화 (Docker 내부에서 실행 시)
            extraction_llm = Ollama(
                model="gemma:2b",
                base_url="http://ollama:11434",
                temperature=0.1  # 일관성 있는 결과를 위해 낮은 temperature
            )

            logger.info("LLM for structured extraction loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            extraction_llm = None

    # 더 다양한 패턴 지원
    patterns = {
        "responsibilities": r"(담당업무|주요업무|업무내용|업무|수행업무|주요\s*업무|담당\s*업무|Job Description|Responsibilities)(.*?)(지원자격|자격요건|필수요건|우대사항|복리후생|근무조건|채용절차|전형절차|기타|$)",
        "qualifications": r"(지원자격|자격요건|필수요건|필수\s*자격|지원\s*자격|Required Qualifications|Requirements)(.*?)(우대사항|우대요건|복리후생|근무조건|채용절차|전형절차|기타|$)",
        "preferences": r"(우대사항|우대요건|우대\s*사항|가산점|Preferred Qualifications|Nice to Have)(.*?)(복리후생|근무조건|근무환경|혜택|채용절차|전형절차|기타|$)",
    }

    for row in iterator:
        text = row.description
        sections = {
            "responsibilities": "",
            "qualifications": "",
            "preferences": "",
        }
        if text:
            text = re.sub(r'\[OCR Text\]:', '', text)
            extracted = False
            for key, pattern in patterns.items():
                match = re.search(pattern, text, re.S)
                if match:
                    section_content = match.group(2).strip()
                    if section_content:
                        sections[key] = section_content
                        extracted = True

            # 정규표현식 실패 시 LLM으로 구조화된 정보 추출
            if not extracted and extraction_llm:
                try:
                    # 프롬프트 생성
                    prompt = f"""다음은 채용 공고 내용입니다. 담당업무, 자격요건, 우대사항을 추출하여 JSON 형식으로 반환하세요.
정보가 없는 항목은 빈 문자열로 남겨두세요.

채용 공고:
{text[:2000]}

JSON 형식으로만 답변하세요:
{{"responsibilities": "담당업무 내용", "qualifications": "자격요건 내용", "preferences": "우대사항 내용"}}

JSON:"""

                    result = extraction_llm.invoke(prompt)

                    # JSON 파싱 시도
                    import json
                    # 결과에서 JSON 부분만 추출
                    json_start = result.find('{')
                    json_end = result.rfind('}') + 1

                    if json_start != -1 and json_end > json_start:
                        json_str = result[json_start:json_end]
                        extracted_data = json.loads(json_str)

                        sections["responsibilities"] = extracted_data.get("responsibilities", "").strip()
                        sections["qualifications"] = extracted_data.get("qualifications", "").strip()
                        sections["preferences"] = extracted_data.get("preferences", "").strip()

                        # 성공 로그는 제거 (너무 많아서)
                    else:
                        # JSON 파싱 실패 시 전체 텍스트 사용
                        sections["responsibilities"] = text
                        import logging
                        logging.getLogger(__name__).warning(f"LLM JSON parsing failed for {row.company}")

                except Exception as e:
                    # LLM 실패 시 전체 텍스트 사용
                    sections["responsibilities"] = text
                    import logging
                    logging.getLogger(__name__).warning(f"LLM extraction error for {row.company}: {e}")
            elif not extracted:
                # LLM 사용 안함 설정이거나 로드 실패 시
                sections["responsibilities"] = text

        yield Row(
            company=row.company,
            cleaned_title=clean_text(row.title),
            responsibilities=sections["responsibilities"],
            qualifications=sections["qualifications"],
            preferences=sections["preferences"],
            link=row.link,
            keyword=row.keyword,
            description=text if text else ""  # 원문 보존
        )

def main(input_dir=None, output_dir=None, use_summarization=None, **kwargs):
    # Airflow에서 전달된 파라미터가 있으면 사용, 없으면 설정 파일에서 로드
    if input_dir is None or output_dir is None or use_summarization is None:
        config = load_config()
        input_dir = input_dir or config['parser_input_dir']
        output_dir = output_dir or config['parser_output_dir']
        use_summarization = use_summarization if use_summarization is not None else config['use_summarization']
    
    spark = SparkSession.builder \
        .appName("CareerRAG Preprocessing") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.ui.showConsoleProgress", "false") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    try:
        # JSON 파일 로드
        df = spark.read.option("multiLine", True).json(f"{input_dir}/*.json")
        print(f"Loaded {df.count()} records")

        # 데이터 정제
        from functools import partial
        process_func = partial(process_partition, use_summarization=use_summarization)
        processed_rdd = df.rdd.mapPartitions(process_func)

        schema = StructType([
            StructField("company", StringType(), True),
            StructField("cleaned_title", StringType(), True),
            StructField("responsibilities", StringType(), True),
            StructField("qualifications", StringType(), True),
            StructField("preferences", StringType(), True),
            StructField("link", StringType(), True),
            StructField("keyword", StringType(), True),
            StructField("description", StringType(), True),  # 원문 필드 추가
        ])

        final_df = spark.createDataFrame(processed_rdd, schema)

        # 출력 디렉토리 생성
        ensure_dir(output_dir)
        
        # 저장
        output_path = f"{output_dir}/processed_postings.parquet"
        final_df.write.mode("overwrite").parquet(output_path)
        print(f"Processed {final_df.count()} records -> {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        spark.stop()

if __name__ == "__main__":
    main()