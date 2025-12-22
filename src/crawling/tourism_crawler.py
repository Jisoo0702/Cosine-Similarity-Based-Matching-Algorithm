"""
VisitKorea(한국관광공사) 관광지 상세정보 크롤러

수집 항목
- 장소명
- 상세정보
- 상세주소
- 해시태그
"""

from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import time


URLS = [
    "https://korean.visitkorea.or.kr/detail/ms_detail.do?cotid=c04d8d19-2acf-4151-a33b-f7a76633f857&big_category=A02&mid_category=A0201&big_area=37",
    "https://korean.visitkorea.or.kr/detail/ms_detail.do?cotid=fa615ceb-d734-4ee5-92d6-c956cd05d4cc&big_category=A02&mid_category=A0201&big_area=37",
    "https://korean.visitkorea.or.kr/detail/ms_detail.do?cotid=5062a741-eaa8-4d9d-a56d-eb4205e4ca8a&big_category=A01&mid_category=A0101&big_area=37",
    "https://korean.visitkorea.or.kr/detail/ms_detail.do?cotid=61035ea7-0b4e-4d86-b1f9-281ce1e05dce&big_category=A05&mid_category=A0502&big_area=37",
    "https://korean.visitkorea.or.kr/detail/ms_detail.do?cotid=9452e074-6a4f-4ea4-b461-96c8ebc779b3&big_category=A05&mid_category=A0502&big_area=37",
    "https://korean.visitkorea.or.kr/detail/ms_detail.do?cotid=55871477-6a47-48b9-bf37-dedf0940814c&big_category=A05&mid_category=A0502&big_area=37",
    "https://korean.visitkorea.or.kr/detail/ms_detail.do?cotid=26b4b63c-1546-4016-8839-63f2847a6bf1&big_category=A05&mid_category=A0502&big_area=37",
    "https://korean.visitkorea.or.kr/detail/ms_detail.do?cotid=ea7b2055-77d8-4131-81bc-b9a10cbfd040&big_category=A02&mid_category=A0201&big_area=37",
    "https://korean.visitkorea.or.kr/detail/ms_detail.do?cotid=e3220f63-7ac7-448a-b70a-590f5f6a4d41&big_category=A02&mid_category=A0201&big_area=37"
]


OUTPUT_FILE = "visitkorea_places.csv"


def setup_driver() -> webdriver.Chrome:
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    # options.add_argument("--headless")   # 서버 사용 시
    return webdriver.Chrome(options=options)


def parse_page(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    place_name = soup.select_one("#topTitle")
    details = soup.select_one("div.inr_wrap p")
    hashtags = soup.select("div.tag_cont span")

    address_spans = soup.select("div.area_txtView.bottom span")
    address = " ".join(span.get_text(strip=True) for span in address_spans)

    return {
        "장소명": place_name.get_text(strip=True) if place_name else "",
        "상세정보": details.get_text(strip=True) if details else "",
        "상세주소": address,
        "해시태그": ", ".join(tag.get_text(strip=True) for tag in hashtags),
    }


def collect_places(driver: webdriver.Chrome) -> list:
    results = []

    for url in URLS:
        driver.get(url)
        time.sleep(4)

        try:
            info = parse_page(driver.page_source)
            results.append(info)
            print(f"수집 완료: {info['장소명']}")
        except Exception as e:
            print(f"오류 발생: {url} / {e}")

    return results


def save_csv(rows: list, path: str) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"CSV 저장 완료: {path}")


def main() -> None:
    driver = setup_driver()
    try:
        rows = collect_places(driver)
        save_csv(rows, OUTPUT_FILE)
    finally:
        driver.quit()
        print("크롤링 종료")


if __name__ == "__main__":
    main()
