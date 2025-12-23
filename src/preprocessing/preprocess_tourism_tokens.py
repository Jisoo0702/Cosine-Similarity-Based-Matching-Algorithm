from pathlib import Path

import pandas as pd
from konlpy.tag import Okt


RAW_DATA_PATH = Path("data/tourism_boryeong_raw.xlsx")
STOPWORD_PATH = Path("resources/stopwords-ko.txt")
OUTPUT_PATH = Path("data/tourism_boryeong_tokens.csv")


# 기본 불용어 + 지역/설명에 자주 등장하는 단어
DEFAULT_STOPWORDS = [
    "그리고",
    "그래서",
    "이지만",
    "그런데",
    "더불어",
    "따라서",
    "오직",
    "예를 들어",
    "또한",
    "하지만",
    "즉",
    "같은",
    "매우",
    "엄청",
    "무척",
    "대단히",
    "굉장히",
    "상당히",
    "정말",
    "참",
    "확실히",
    "틀림없이",
    "완전히",
    "절대적으로",
    "너무",
    "아주",
    "많이",
    "더더욱",
    "더욱",
    "훨씬",
    "비교적",
    "꽤",
    "지나치게",
    "살짝",
    "살며시",
    "분명히",
    "의심의 여지 없이",
    "반드시",
    "꼭",
    "극도로",
    "차마",
    "도저히",
    "만약",
    "과연",
    "어찌나",
    "굳이",
    "간신히",
    "그나마",
    "그토록",
    "이토록",
    "과하게",
    "너무나",
    "참으로",
    "기필코",
    "완벽하게",
    "필히",
    "결코",
    "완벽한",
    "자나친",
    "과한",
    # 지역/관광 설명에 자주 등장하는 일반 단어
    "충청남도",
    "보령",
    "보령시",
    "호선",
    "고속도로",
    "건물",
    "주변",
    "통해",
    "위치",
    "메뉴",
    "규모",
    "국도",
    "시설",
    "자리",
    "인근",
    "접근",
    "타고",
    "위해",
]


def load_stopwords() -> set[str]:
    """기본 불용어 + 파일에서 읽은 불용어를 합쳐서 반환."""
    extra = []
    if STOPWORD_PATH.exists():
        with STOPWORD_PATH.open("r", encoding="utf-8") as f:
            extra = [line.strip() for line in f if line.strip()]

    return set(DEFAULT_STOPWORDS + extra)


def build_custom_keywords(df: pd.DataFrame) -> list[str]: # 관광지 이름 컬럼(name)을 사용자 정의 키워드로 사용
    return df["name"].dropna().astype(str).tolist()


def preprocess_text(
    text: str, custom_keywords: list[str], stopwords: set[str], okt: Okt
) -> list[str]: # 관광지 설명에서 키워드(관광지명 + 명사)를 추출
    if pd.isna(text):
        return []

    text = str(text)
    extracted = []

    # 관광지 이름 먼저 추출
    for kw in custom_keywords:
        if kw in text:
            extracted.append(kw)
            text = text.replace(kw, "")

    # 명사 추출
    nouns = okt.nouns(text)

    # 불용어, 1글자 단어 제거
    filtered = [w for w in nouns if w not in stopwords and len(w) > 1]

    return extracted + filtered

# 토큰화
def main() -> None:
    df = pd.read_excel(RAW_DATA_PATH)

    okt = Okt()
    stopwords = load_stopwords()
    custom_keywords = build_custom_keywords(df)

    df["tokens"] = df["detailed_info"].apply(
        lambda x: preprocess_text(x, custom_keywords, stopwords, okt)
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"전처리 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
