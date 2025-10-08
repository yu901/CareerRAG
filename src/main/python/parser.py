import findspark
findspark.init()
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, StringType
import re
from src.main.python.utils import load_config, ensure_dir, clean_text


def process_partition(iterator, use_summarization=True):
    # 파티션당 한 번만 모델 로딩
    summarization_chain = None
    if use_summarization:
        try:
            import warnings
            import logging
            import os

            # 경고 메시지 억제
            warnings.filterwarnings('ignore')
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            logging.getLogger("transformers").setLevel(logging.ERROR)

            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
            from langchain_huggingface.llms import HuggingFacePipeline
            from langchain_core.prompts import PromptTemplate

            model_name = "gogamza/kobart-summarization"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, ignore_mismatched_sizes=True)
            summarization_pipeline = pipeline(
                "summarization",
                model=model,
                tokenizer=tokenizer,
                max_length=200,
                min_length=30,
                do_sample=False
            )
            llm = HuggingFacePipeline(pipeline=summarization_pipeline)
            prompt_template = """Summarize the following Korean job posting:\n\n        {text}\n\n        Summary:"""
            prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
            summarization_chain = prompt | llm
        except Exception:
            summarization_chain = None

    patterns = {
        "responsibilities": r"(담당업무|주요업무|업무내용)(.*?)(지원자격|자격요건|우대사항|복리후생|근무조건|채용절차|$)",
        "qualifications": r"(지원자격|자격요건|필수요건)(.*?)(우대사항|복리후생|근무조건|채용절차|$)",
        "preferences": r"(우대사항)(.*?)(복리후생|근무조건|채용절차|$)",
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

            if not extracted and summarization_chain:
                try:
                    summary = summarization_chain.invoke({"text": text})
                    sections["responsibilities"] = summary
                except Exception:
                    sections["responsibilities"] = text
            elif not extracted:
                sections["responsibilities"] = text

        yield Row(
            company=row.company,
            cleaned_title=clean_text(row.title),
            responsibilities=sections["responsibilities"],
            qualifications=sections["qualifications"],
            preferences=sections["preferences"],
            link=row.link,
            keyword=row.keyword
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