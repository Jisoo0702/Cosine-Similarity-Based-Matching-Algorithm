"""
Playboard(국내 여행)에서 유튜브 채널 정보를 수집하는 간단한 크롤러.

수집 항목
- 채널명
- 채널 설명
- 해시태그
- 구독자 수 및 영상 개수
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
import csv
import time


PLAYBOARD_URL = (
    "https://playboard.co/search"
    "?q=%EA%B5%AD%EB%82%B4%EC%97%AC%ED%96%89&country=KR&sortTypeId=1"
)

MAX_CHANNELS = 1500
SCROLL_PAUSE = 1.5
MAX_SCROLL_ATTEMPTS = 25
OUTPUT_FILE = "youtube_channels_1500.csv"


def setup_driver() -> webdriver.Chrome:
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    # options.add_argument("--headless")  # 서버에서 돌릴 때는 주석 해제
    driver = webdriver.Chrome(options=options)
    return driver


def parse_channel_meta(meta) -> list:
    """채널 카드 하나에서 텍스트 정보만 뽑아온다."""
    # 채널명
    channel_name = meta.find_element(By.XPATH, './h2[@class="name"]/a').text.strip()

    # 채널 설명
    try:
        channel_desc = meta.find_element(
            By.XPATH, './div[@class="desc"]'
        ).text.strip()
    except Exception:
        channel_desc = ""

    # 해시태그
    try:
        hashtag_elements = meta.find_elements(
            By.XPATH, './ul[@class="keywords ttags"]/li/a'
        )
        hashtags = ", ".join(tag.text.strip() for tag in hashtag_elements) or ""
    except Exception:
        hashtags = ""

    # 구독자 수 / 영상 개수
    try:
        scores_elements = meta.find_elements(
            By.XPATH, './ul[@class="simple-scores"]/li'
        )
        scores = ", ".join(score.text.strip() for score in scores_elements) or ""
    except Exception:
        scores = ""

    return [channel_name, channel_desc, hashtags, scores]


def collect_channels(driver: webdriver.Chrome) -> list:
    data = []
    collected = 0
    scroll_attempts = 0

    print("크롤링 시작")

    while collected < MAX_CHANNELS and scroll_attempts < MAX_SCROLL_ATTEMPTS:
        meta_elements = driver.find_elements(
            By.XPATH, '//div[contains(@class, "meta")]'
        )

        for meta in meta_elements:
            if collected >= MAX_CHANNELS:
                break

            try:
                row = parse_channel_meta(meta)
            except Exception as e:
                print(f"채널 정보 파싱 중 오류: {e}")
                continue

            if row not in data:  # 단순 중복 제거
                data.append(row)
                collected += 1
                print(f"[{collected}] {row[0]}")

        # 더 스크롤할지 결정
        last_height = driver.execute_script("return document.body.scrollHeight")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE)
        new_height = driver.execute_script("return document.body.scrollHeight")

        if new_height == last_height:
            scroll_attempts += 1
            print(f"새 콘텐츠 없음 ({scroll_attempts}/{MAX_SCROLL_ATTEMPTS})")
        else:
            scroll_attempts = 0

    print(f"총 {len(data)}개 채널 수집 완료")
    return data


def save_to_csv(rows: list, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["채널명", "채널설명", "해시태그", "구독자 수 및 영상 개수"])
        writer.writerows(rows)
    print(f"CSV 저장 완료: {path}")


def main() -> None:
    driver = setup_driver()
    driver.get(PLAYBOARD_URL)
    time.sleep(3)

    try:
        rows = collect_channels(driver)
        save_to_csv(rows, OUTPUT_FILE)
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
