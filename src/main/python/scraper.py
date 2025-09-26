import requests
from bs4 import BeautifulSoup
import json
import time
import os

BASE_URL = "https://www.saramin.co.kr/zf_user/search/recruit"

def get_job_postings(keyword: str, num_pages: int = 10):
    """
    Args:
        keyword (str): 검색할 키워드 (예: "데이터 엔지니어").
        num_pages (int): 스크래핑할 페이지 수.

    Returns:
        list: 채용 공고 정보가 담긴 딕셔너리 리스트.
    """
    job_postings = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    for page in range(1, num_pages + 1):
        params = {
            "searchword": keyword,
            "recruitPage": page
        }
        try:
            response = requests.get(BASE_URL, params=params, headers=headers)
            response.raise_for_status()  # HTTP 오류 발생 시 예외 처리
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # 스크래핑 로직 (웹사이트 구조 변경 시 수정 필요)
            for item in soup.select(".item_recruit"):
                try:
                    title = item.select_one(".job_tit a")["title"]
                    company = item.select_one(".corp_name a").get_text(strip=True)
                    link = "https://www.saramin.co.kr" + item.select_one(".job_tit a")["href"]

                    job_postings.append({
                        "title": title,
                        "company": company,
                        "link": link,
                        "keyword": keyword
                    })
                except (AttributeError, TypeError, KeyError):
                    # 개별 공고 스크래핑 실패 시 건너뛰기
                    continue
            
            print(f"Scraped page {page} for keyword '{keyword}'. Found {len(job_postings)} postings so far.")
            
            # 서버 부하 방지
            time.sleep(1)

        except requests.exceptions.RequestException as e:
            print(f"Error scraping page {page} for keyword '{keyword}': {e}")
            continue
            
    return job_postings

if __name__ == "__main__":
    # 데이터 저장 디렉토리 생성
    if not os.path.exists("data"):
        os.makedirs("data")

    search_keyword = "데이터엔지니어"
    postings = get_job_postings(search_keyword, num_pages=10) # 테스트 목적 페이지 제한 (10페이지)

    # 수집 데이터 JSON 파일 저장
    file_path = f"data/{search_keyword}_postings.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(postings, f, ensure_ascii=False, indent=4)

    print(f"Successfully scraped {len(postings)} job postings for '{search_keyword}'.")
    print(f"Data saved to {file_path}")
