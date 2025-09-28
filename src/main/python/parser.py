import findspark
findspark.init()
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType
import re

def clean_text(text: str) -> str:
    """텍스트 정제 (불필요한 문자, 공백, HTML 태그 제거)."""
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def process_partition(iterator):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    from langchain_huggingface.llms import HuggingFacePipeline
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnableSequence

    model_name = "gogamza/kobart-summarization"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        summarization_pipeline = pipeline(
            "summarization",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            min_length=30,
            do_sample=False
        )
        llm = HuggingFacePipeline(pipeline=summarization_pipeline)
        prompt_template = """Summarize the following Korean job posting:\n\n        {text}\n\n        Summary:"""
        prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
        summarization_chain = prompt | llm
    except Exception as e:
        # If model loading fails, proceed without summarization
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
                except Exception as e:
                    sections["responsibilities"] = text # Fallback to full text

            elif not extracted:
                sections["responsibilities"] = text # Fallback to full text


        yield Row(
            company=row.company,
            cleaned_title=clean_text(row.title),
            responsibilities=sections["responsibilities"],
            qualifications=sections["qualifications"],
            preferences=sections["preferences"],
            link=row.link,
            keyword=row.keyword
        )

def main():
    """Spark 세션 초기화 및 데이터 전처리 수행."""
    spark = SparkSession.builder \
        .appName("CareerRAG Preprocessing") \
        .master("local[*]") \
        .getOrCreate()

    # 원천 데이터 로드 (scraper.py 저장 파일)
    raw_data_path = "data/데이터엔지니어_postings.json"
    try:
        df = spark.read.option("multiLine", True).json(raw_data_path)
    except Exception as e:
        print(f"Error reading data from {raw_data_path}: {e}")
        spark.stop()
        return

    print("Raw data schema:")
    df.printSchema()
    df.show(5, truncate=False)

    # 데이터 정제
    processed_rdd = df.rdd.mapPartitions(process_partition)

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

    print("Processed data schema:")
    final_df.printSchema()
    print("Sample of processed data:")
    final_df.show(5, truncate=False)

    # 처리된 데이터 저장
    output_path = "data/processed_postings.parquet"
    final_df.write.mode("overwrite").parquet(output_path)

    print(f"Processed data saved to {output_path}")

    spark.stop()

if __name__ == "__main__":
    main()
