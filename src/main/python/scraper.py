import requests
from bs4 import BeautifulSoup
import json
import time
import os
from urllib.parse import urlparse, parse_qs
from PIL import Image
import pytesseract
from io import BytesIO
import warnings
from urllib3.exceptions import InsecureRequestWarning
warnings.filterwarnings("ignore", category=InsecureRequestWarning)

BASE_URL = "https://www.saramin.co.kr/zf_user/search/recruit"

COMMON_IMAGE_PATTERNS = [
    "static.saraminimage.co.kr/static/hiring/images/template/",
    "cdn.jumpit.co.kr/jumpit/bnr_sri_position_renew.png"
]

def get_job_details(job_link: str, headers: dict) -> str:
    """
    채용 공고 상세 페이지에서 내용을 스크래핑합니다.
    """
    try:
        response = requests.get(job_link, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # 상세 내용이 포함된 div를 찾습니다.
        # HTML 구조에 따라 선택자를 변경해야 합니다.
        detail_content_div = soup.select_one("div.user_content")
        if detail_content_div:
            # 모든 텍스트를 추출하고 불필요한 공백을 제거합니다.
            # <br> 태그를 개행 문자로 변환하여 텍스트 구조를 유지합니다.
            for br in detail_content_div.find_all("br"):
                br.replace_with("\n")
            
            content = detail_content_div.get_text(separator=" ", strip=True)

            # 이미지에서 텍스트 추출 (OCR)
            for img in detail_content_div.find_all("img"):
                img_url = img.get("src")
                if img_url and img_url.startswith("http"): # 유효한 이미지 URL인 경우
                    # 이미지 URL 정제 (중복된 도메인 제거 및 saraminimage.co.kr?url= 패턴 처리)
                    if img_url.count("https://pds.saramin.co.kr/") > 1:
                        img_url = img_url.replace("https://pds.saramin.co.kr/", "", 1)
                        if not img_url.startswith("http"):
                            img_url = "https://pds.saramin.co.kr/" + img_url
                    elif "https://www.saraminimage.co.kr?url=" in img_url:
                        img_url = img_url.split("https://www.saraminimage.co.kr?url=")[1]
                        # Ensure it's a full URL, if not, prepend http://
                        if not img_url.startswith("http"):
                            img_url = "http://" + img_url

                    # 공통 이미지 패턴에 해당하는 경우 OCR 건너뛰기
                    if any(pattern in img_url for pattern in COMMON_IMAGE_PATTERNS):
                        continue
                    try:
                        img_response = requests.get(img_url, headers=headers, timeout=5, verify=False) # SSL 검증 무시
                        img_response.raise_for_status()

                        # Content-Type 헤더 확인
                        if 'content-type' in img_response.headers and 'image' in img_response.headers['content-type']:
                            img_data = Image.open(BytesIO(img_response.content))
                            ocr_text = pytesseract.image_to_string(img_data, lang='kor+eng') # 한국어+영어 OCR
                            if ocr_text.strip():
                                content += f"\n[OCR Text]: {ocr_text.strip()}"
                        else:
                            print(f"DEBUG: URL {img_url} did not return an image (Content-Type: {img_response.headers.get('content-type')})")

                    except Image.UnidentifiedImageError:
                        print(f"Error performing OCR on image {img_url}: Cannot identify image file (possibly not an image or corrupted).")
                    except Exception as e:
                        print(f"Error performing OCR on image {img_url}: {e}")
            
            return content
        else:
            return "상세 내용을 찾을 수 없습니다."

    except Exception as e:
        print(f"Error processing job detail page {job_link}: {e}")
        return "상세 내용 처리 오류."

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

                    # rec_idx 추출 및 상세 페이지 URL 생성
                    parsed_link = urlparse(link)
                    query_params = parse_qs(parsed_link.query)
                    rec_idx = query_params.get('rec_idx', [''])[0]
                    
                    detail_link = f"https://www.saramin.co.kr/zf_user/jobs/relay/view-detail?rec_idx={rec_idx}"

                    # 상세 페이지 스크래핑
                    job_description = get_job_details(detail_link, headers)
                    time.sleep(0.5) # 상세 페이지 요청 간 지연

                    job_postings.append({
                        "title": title,
                        "company": company,
                        "link": link,
                        "keyword": keyword,
                        "description": job_description
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
