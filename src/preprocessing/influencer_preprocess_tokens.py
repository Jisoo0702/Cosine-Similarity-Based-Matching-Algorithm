from pathlib import Path
import re
import pandas as pd
from konlpy.tag import Okt


RAW_DATA_PATH = Path("data/influencer_raw.xlsx")    
STOPWORD_PATH = Path("resources/stopwords-ko.txt")
OUTPUT_PATH = Path("data/influencer_tokens.csv")


# 기본 불용어 + 지역/감탄, 이모티콘류
BASE_STOPWORDS = [
    "그리고", "그래서", "이지만", "그런데", "더불어", "따라서", "오직", "사이", "넓고", "없고", "상세", "너무",
    "중에", "가지고", "예를", "들어", "또한", "하지만", "즉", "같은", "매우", "엄청", "무척", "갖춘", "했던",
    "주셔서", "그닥", "넘", "좋고", "이곳", "대단히", "굉장히", "상당히", "정말", "참", "확실히", "틀림없이",
    "있을까", "있어요", "보이는", "가지않아도", "이어", "암튼", "곳", "완전히", "절대적으로", "아주", "많이",
    "더더욱", "더욱", "훨씬", "이상", "로", "물론", "너무많이", "저는", "너무너무", "통해", "사랑일까",
    "비교적", "꽤", "지나치게", "살짝", "살며시", "분명히", "의심의", "여지", "없이", "여긴", "그동안",
    "맞춘", "1층", "위치한", "내가", "등", "등등", "반드시", "꼭", "극도로", "차마", "도저히", "만약",
    "과연", "어찌나", "굳이", "드디어", "될리", "없구요", "그냥", "아니라", "주는", "사실", "비해",
    "간신히", "그나마", "그토록", "이토록", "과하게", "너무나", "참으로", "기필코", "완벽하게", "필히",
    "결코", "워낙", "와서", "하는", "한번", "이곳은", "많고", "완벽한", "자나친", "과한", "또", "다시",
    "더", "있다", "없다", "가다", "보다", "하다", "이다", "이것이", "와", "제대로", "따로", "생각보다",
    "있었는데", "있는데", "그", "수", "당", "이것", "다", "오늘", "그저", "하면", "없다", "자",
    "세", "응", "아니오", "아니요", "모든", "제가", "전", "요즘", "이렇게", "이제", "저의", "오늘은",
    "진짜", "가장", "같아요", "하고", "잘", "좋은", "완전", "그래도", "일단", "있고", "거의", "모두",
    "많은", "있어서", "제", "머", "딱", "있습니다", "이거", "근데", "오랜만에", "되는", "사람이",
    # 감탄/이모티콘류
    "ㅠㅠ", "ㅜㅜ", "ㅎ", "ㅋ", "크크", "하하",
]


# 불용어 처리
def load_stopwords() -> set[str]:
    extra = []
    if STOPWORD_PATH.exists():
        with STOPWORD_PATH.open("r", encoding="utf-8") as f:
            extra = [line.strip() for line in f if line.strip()]

    return set(BASE_STOPWORDS + extra)


# 토큰화
def preprocess_and_tokenize(text: str, stopwords: set[str], okt: Okt) -> list[str]:
    if pd.isna(text):
        return []

    text = str(text)

    # 개행, 특수문자, 중복 감탄 등 정리
    text = text.replace("\n", " ").strip()
    text = re.sub(r"[~♡!?.\[\]{}<>]", "", text)
    text = re.sub(r"[^\w\sㄱ-ㅎㅏ-ㅣ가-힣]", " ", text)
    text = re.sub(r"([ㅋㅎ])\1+", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    # 명사 추출
    nouns = okt.nouns(text)

    # 불용어 제거 + 영어/숫자 포함 토큰 제거 + 1글자 제거
    tokens = [
        w
        for w in nouns
        if w not in stopwords and not re.search(r"[a-zA-Z0-9]", w) and len(w) > 1
    ]

    return tokens


# 토큰화
def main() -> None:
    # 엑셀에서 시트 단위로 읽기
    df_info = pd.read_excel(RAW_DATA_PATH, sheet_name="유튜버 정보")
    df_content = pd.read_excel(RAW_DATA_PATH, sheet_name="콘텐츠 정보")

    okt = Okt()
    stopwords = load_stopwords()

    rows = []

    # 채널별로 묶어서 처리
    for channel in df_content["채널명"].dropna().unique():
        info_part = df_info[df_info["채널명"] == channel]
        content_part = df_content[df_content["채널명"] == channel].copy()

        # 제목 + 설명 + 해시태그 결합
        content_part["combined"] = (
            content_part["동영상 제목"].fillna("")
            + " "
            + content_part["동영상 설명글"].fillna("")
            + " "
            + content_part["해시태그"].fillna("")
        )

        # 채널 설명 + 콘텐츠 텍스트 합치기
        combined_texts = (
            info_part["채널 설명글"].dropna().tolist()
            + content_part["combined"].dropna().tolist()
        )

        for text in combined_texts:
            tokens = preprocess_and_tokenize(text, stopwords, okt)
            rows.append(
                {
                    "channel": channel,
                    "original_text": text,
                    "tokens": tokens,
                }
            )

    df_tokens = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_tokens.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"인플루언서 토큰화 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
