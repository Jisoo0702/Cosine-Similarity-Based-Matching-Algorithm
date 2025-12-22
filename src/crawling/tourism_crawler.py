from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import time

# URL 목록
urls = [ "https://korean.visitkorea.or.kr/detail/ms_detail.do?cotid=c04d8d19-2acf-4151-a33b-f7a76633f857&big_category=A02&mid_category=A0201&big_area=37",
        "https://korean.visitkorea.or.kr/detail/ms_detail.do?cotid=fa615ceb-d734-4ee5-92d6-c956cd05d4cc&big_category=A02&mid_category=A0201&big_area=37",
         "https://korean.visitkorea.or.kr/detail/ms_detail.do?cotid=5062a741-eaa8-4d9d-a56d-eb4205e4ca8a&big_category=A01&mid_category=A0101&big_area=37",
         "https://korean.visitkorea.or.kr/detail/ms_detail.do?cotid=61035ea7-0b4e-4d86-b1f9-281ce1e05dce&big_category=A05&mid_category=A0502&big_area=37",
         "https://korean.visitkorea.or.kr/detail/ms_detail.do?cotid=9452e074-6a4f-4ea4-b461-96c8ebc779b3&big_category=A05&mid_category=A0502&big_area=37",
         "https://korean.visitkorea.or.kr/detail/ms_detail.do?cotid=55871477-6a47-48b9-bf37-dedf0940814c&big_category=A05&mid_category=A0502&big_area=37",
         "https://korean.visitkorea.or.kr/detail/ms_detail.do?cotid=26b4b63c-1546-4016-8839-63f2847a6bf1&big_category=A05&mid_category=A0502&big_area=37",
         "https://korean.visitkorea.or.kr/detail/ms_detail.do?cotid=ea7b2055-77d8-4131-81bc-b9a10cbfd040&big_category=A02&mid_category=A0201&big_area=37",
         "https://korean.visitkorea.or.kr/detail/ms_detail.do?cotid=e3220f63-7ac7-448a-b70a-590f5f6a4d41&big_category=A02&mid_category=A0201&big_area=37"


         ]

# 데이터 저장을 위한 리스트
data = []

# Chrome 드라이버 설정
driver = webdriver.Chrome()

for url in urls:
    driver.get(url)
    time.sleep(5)  # 페이지가 완전히 로드될 때까지 대기

    # 페이지 소스 가져오기
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    try:
        # 장소명
        place_name = soup.select_one("#topTitle").get_text(strip=True) if soup.select_one("#topTitle") else "N/A"

        # 상세정보
        details = soup.select_one("div.inr_wrap p").get_text(strip=True) if soup.select_one("div.inr_wrap p") else "N/A"

        # 상세주소: area_txtView 내의 모든 <span> 요소를 찾아 연결
        address = " ".join([span.get_text(strip=True) for span in soup.select("div.area_txtView.bottom span")])

        # 해시태그
        hashtags = ", ".join([tag.get_text(strip=True) for tag in soup.select("div.tag_cont span")])

        # 데이터 저장
        data.append({
            "장소명": place_name,
            "상세정보": details,
            "상세주소": address,
            "해시태그": hashtags
        })

    except Exception as e:
        print(f"Error occurred for URL: {url} - {e}")

# DataFrame으로 변환
df = pd.DataFrame(data)

# CSV 파일로 저장
df.to_csv("places김제.csv", index=False, encoding="utf-8-sig")

# 드라이버 종료
driver.quit()

print("크로링 완료.")
