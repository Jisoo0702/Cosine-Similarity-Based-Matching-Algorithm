from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


RAW_DATA_PATH = Path("data/influencer_tokens.csv")  # 전처리(토큰화) 결과
OUTPUT_PATH = Path("data/influencer_ctfidf_summary.csv")


# 기본 불용어 + 제거/대체 규칙에 자주 등장하는 단어
REMOVE_WORDS = [
    "링드", "마켓", "스타", "모든", "채널", "포스", "스린", "만행", "에서", "치기", "철행", "상촬", "진촬", "인트", "주도",
    "마무리", "목차", "코리", "잔샘", "으로", "안녕", "시행", "유카", "병맛", "베스트",
    "봉선", "쿠카", "습니다", "손질", "브레", "상필", "활동", "선물", "산시", "대표", "무장애", "도보", "접근", "명소",
    "복화술", "별별", "연소", "독기", "주소", "클릭", "공복", "브랜드", "안경", "카린", "플루", "프라", "마카", "파타",
    "인스타", "쇼츠", "볼품", "잔비", "폴킴", "우러스", "불능", "주차장", "상대", "언트", "케릭", "무삭", "뿡뿡", "대회",
    "미니", "행유", "병맛", "활용",
]

REPLACE_WORDS = {
    "컬리": "마켓컬리",
    "레길": "둘레길",
    "비엔": "비엔나",
    "른견": "어른견",
    "집트": "이집트",
    "국행": "국내여행",
    "메디치": "이탈리아",
    "버틀러": "집사",
    "웅천": "여수",
    "소도": "소도시",
    "혼행": "신혼여행",
    "름밤": "여름밤",
}

# 불용어 처리
STOP_KEYWORDS = ["사우스", "나크", "시작"]


def _ensure_tokens_list(x) -> list[str]:
    """
    tokens 컬럼이 리스트로 들어오는 걸 전제로 하지만,
    csv 저장 과정에서 문자열로 들어오는 경우를 대비해 간단히 처리.
    """
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    s = str(x).strip()

    # 예: "['a', 'b']" 형태로 들어온 경우(완벽 파싱이 필요하면 ast.literal_eval 추천)
    if s.startswith("[") and s.endswith("]"):
        s = s.strip("[]")
        parts = [p.strip().strip("'").strip('"') for p in s.split(",")]
        return [p for p in parts if p]

    # 예: "a b c" 형태
    return [t for t in s.split() if t]


def preprocess_tokens(tokens: list[str]) -> list[str]:
    cleaned = []
    for t in tokens:
        if not t or len(t) <= 1:
            continue
        if t in REMOVE_WORDS:
            continue
        t = REPLACE_WORDS.get(t, t)
        cleaned.append(t)
    return cleaned


def extract_top_keywords(c_tfidf_matrix, vectorizer, n_top_keywords: int = 10) -> list[list[str]]:
    feature_names = vectorizer.get_feature_names_out()
    keywords_per_cluster = []

    for cluster_idx in range(c_tfidf_matrix.shape[0]):
        row = c_tfidf_matrix[cluster_idx].toarray()[0]
        top_idx = row.argsort()[-n_top_keywords:][::-1]
        keywords_per_cluster.append([feature_names[i] for i in top_idx])

    return keywords_per_cluster


def process_channel(channel_name: str, channel_docs_tokens: list[list[str]]) -> dict:
    df_ch = pd.DataFrame({"tokens": channel_docs_tokens})
    df_ch["tokens"] = df_ch["tokens"].apply(preprocess_tokens)
    df_ch["joined_tokens"] = df_ch["tokens"].apply(lambda x: " ".join(x))

    # 문서 수가 너무 적으면 클러스터링 스킵
    num_docs = len(df_ch)
    num_clusters = min(5, num_docs)
    if num_clusters < 2:
        return {
            "channel": channel_name,
            "best_cluster": None,
            "top_keywords": "데이터 부족",
            "num_docs": num_docs,
        }

    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(df_ch["joined_tokens"])

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
    df_ch["cluster"] = kmeans.fit_predict(tfidf_matrix)

    # C-TF-IDF
    cluster_texts = df_ch.groupby("cluster")["joined_tokens"].apply(lambda x: " ".join(x))
    cluster_tfidf = vectorizer.fit_transform(cluster_texts)
    c_tfidf = normalize(cluster_tfidf, norm="l1", axis=1)

    top_keywords = extract_top_keywords(c_tfidf, vectorizer, n_top_keywords=10)

    # 불필요 키워드 제거 + 중복 제거
    cleaned_keywords = []
    for kws in top_keywords:
        uniq = list(dict.fromkeys(kws))  # 순서 유지 중복 제거
        uniq = [w for w in uniq if w not in STOP_KEYWORDS]
        cleaned_keywords.append(uniq)

    # 가장 관련 있는 클러스터: 문서 수가 가장 많은 클러스터
    best_cluster = df_ch["cluster"].value_counts().idxmax()
    best_keywords = cleaned_keywords[best_cluster] if best_cluster < len(cleaned_keywords) else []

    return {
        "channel": channel_name,
        "best_cluster": int(best_cluster),
        "top_keywords": ", ".join(best_keywords),
        "num_docs": num_docs,
    }


# 토큰화
def main() -> None:
    df = pd.read_csv(RAW_DATA_PATH, encoding="utf-8-sig")

    # 기대 컬럼: channel, tokens
    # (네가 전처리에서 저장한 컬럼명이 다르면 여기만 바꾸면 됨)
    if "channel" not in df.columns or "tokens" not in df.columns:
        raise ValueError("CSV에 'channel'과 'tokens' 컬럼이 필요합니다.")

    df["tokens"] = df["tokens"].apply(_ensure_tokens_list)

    summaries = []
    for channel_name, g in df.groupby("channel"):
        docs_tokens = g["tokens"].tolist()  # 문서 단위 tokens 리스트
        summaries.append(process_channel(channel_name, docs_tokens))

    out = pd.DataFrame(summaries).sort_values(["num_docs"], ascending=False)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"모델링 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
