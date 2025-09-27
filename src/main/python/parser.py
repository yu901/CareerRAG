import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
import re

def clean_text(text: str) -> str:
    """텍스트 정제 (불필요한 문자, 공백, HTML 태그 제거)."""
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def main():
    """Spark 세션 초기화 및 데이터 전처리 수행."""
    spark = SparkSession.builder \
        .appName("CareerRAG Preprocessing") \
        .master("local[*]") \
        .getOrCreate()

    # UDF 등록
    clean_text_udf = udf(clean_text, StringType())

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

    # 데이터 정제 (title 및 description 컬럼에 UDF 적용)
    processed_df = df.withColumn("cleaned_title", clean_text_udf(col("title")))\
                     .withColumn("cleaned_description", clean_text_udf(col("description")))
    
    # 필요 컬럼 선택
    final_df = processed_df.select("company", "cleaned_title", "cleaned_description", "link", "keyword")

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
