from pathlib import Path
import ast

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from gensim import corpora
from gensim.models import LdaModel


TOKENS_DATA_PATH = Path("data/influencer_tokens.csv")  # influencer_preprocess_tokens.py 결과
OUTPUT_TOPICS_PATH = Path("data/modeling/influencer_topics.csv")
OUTPUT_DOC_TOPICS_PATH = Path("data/modeling/influencer_doc_topics.csv")


# 단어 제거/대체 규칙
REMOVE_WORDS = [
    "링드", "마켓", "스타", "모든", "채널", "포스", "스린", "만행", "에서", "치기", "철행", "상촬", "진촬", "인트", "주도",
    "마무리", "목차", "코리", "잔샘", "으로", "안녕", "시행", "유카", "병맛", "베스트", "봉선", "쿠카", "습니다", "손질",
    "브레", "상필", "활동", "선물", "산시", "대표", "무장애", "도보", "접근", "명소", "복화술", "별별", "연소", "독기",
    "주소", "클릭", "공복", "브랜드", "안경", "카린", "플루", "프라", "마카", "파타", "인스타", "쇼츠", "볼품", "잔비",
    "폴킴", "우러스", "불능", "주차장", "상대", "언트", "케릭", "무삭", "뿡뿡", "대회", "미니", "행유", "활용",
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

STOP_KEYWORDS = ["사우스", "나크", "시작"]


# 토큰 정리(제거/대체/길이)
def clean_tokens(tokens: list[str]) -> list[str]:
    cleaned = []
    for t in tokens:
        t = REPLACE_WORDS.get(t, t)
        if t in REMOVE_WORDS:
            continue
        if len(t) <= 1:
            continue
        cleaned.append(t)
    return cleaned


# TF-IDF + 클러스터링
def build_clusters(df: pd.DataFrame, num_clusters: int = 5, max_features: int = 1000) -> pd.DataFrame:
    df = df.copy()
    df["tokens"] = df["tokens"].apply(clean_tokens)
    df["joined_tokens"] = df["tokens"].apply(lambda x: " ".join(x))

    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(df["joined_tokens"])

    # 문서 수가 너무 적으면 클러스터 수 축소
    n = min(num_clusters, len(df))
    if n < 2:
        df["cluster"] = 0
        return df

    kmeans = KMeans(n_clusters=n, random_state=42)
    df["cluster"] = kmeans.fit_predict(tfidf_matrix)

    return df


# C-TF-IDF
def compute_c_tfidf(df: pd.DataFrame, max_features: int = 1000):
    cluster_texts = df.groupby("cluster")["joined_tokens"].apply(lambda x: " ".join(x))

    vectorizer = TfidfVectorizer(max_features=max_features)
    cluster_tfidf_matrix = vectorizer.fit_transform(cluster_texts)

    c_tfidf_matrix = normalize(cluster_tfidf_matrix, norm="l1", axis=1)
    feature_names = vectorizer.get_feature_names_out()

    return c_tfidf_matrix, feature_names


# 클러스터별 상위 키워드 추출
def extract_top_keywords(c_tfidf_matrix, feature_names, n_top_keywords: int = 10) -> list[list[str]]:
    top_keywords = []
    for cluster_idx in range(c_tfidf_matrix.shape[0]):
        row = c_tfidf_matrix[cluster_idx].toarray()[0]
        top_indices = row.argsort()[-n_top_keywords:][::-1]
        kws = [feature_names[i] for i in top_indices]
        kws = [k for k in dict.fromkeys(kws) if k not in STOP_KEYWORDS]  # 중복 제거 + 불필요 단어 제거
        top_keywords.append(kws)
    return top_keywords


# LDA 토픽 모델링
def train_lda(df: pd.DataFrame, num_topics: int = 5, passes: int = 10):
    grouped_tokens = df.groupby("cluster")["tokens"].sum().reset_index()

    dictionary = corpora.Dictionary(grouped_tokens["tokens"])
    corpus = [dictionary.doc2bow(text) for text in grouped_tokens["tokens"]]

    lda_model = LdaModel(
        corpus=corpus,
        num_topics=num_topics,
        id2word=dictionary,
        passes=passes,
        random_state=42,
    )

    return lda_model, corpus, dictionary, grouped_tokens


# 문서별 토픽 확률 + 대표 토픽 + 키워드
def get_doc_topics(lda_model: LdaModel, corpus, num_topics: int):
    dominant_topics = []
    topic_probs_all = []
    keywords_all = []

    for doc in corpus:
        topic_probs = lda_model.get_document_topics(doc, minimum_probability=0.0)
        topic_probs_sorted = [prob for _, prob in sorted(topic_probs, key=lambda x: x[0])]

        dominant = max(topic_probs, key=lambda x: x[1])[0]
        dominant_topics.append(dominant)
        topic_probs_all.append(topic_probs_sorted)

        keywords = [w for w, _ in lda_model.show_topic(dominant, topn=10)]
        keywords_all.append(keywords)

    topics_df = pd.DataFrame(topic_probs_all, columns=[f"Topic_{i}" for i in range(num_topics)])
    topics_df["dominant_topic"] = dominant_topics
    topics_df["keywords"] = keywords_all

    return topics_df


# 저장
def save_results(lda_model: LdaModel, num_topics: int, topics_df: pd.DataFrame):
    OUTPUT_TOPICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    topic_rows = []
    for topic_id in range(num_topics):
        words = [w for w, _ in lda_model.show_topic(topic_id, topn=10)]
        topic_rows.append({"topic_id": topic_id, "keywords": ", ".join(words)})

    pd.DataFrame(topic_rows).to_csv(OUTPUT_TOPICS_PATH, index=False, encoding="utf-8-sig")
    topics_df.to_csv(OUTPUT_DOC_TOPICS_PATH, index=False, encoding="utf-8-sig")

    print(f"토픽 키워드 저장 완료: {OUTPUT_TOPICS_PATH}")
    print(f"토픽 확률/대표 토픽 저장 완료: {OUTPUT_DOC_TOPICS_PATH}")


# 실행
def main() -> None:
    df = pd.read_csv(TOKENS_DATA_PATH, encoding="utf-8-sig")

    # tokens가 문자열로 저장되어 있을 경우(list로 복원)
    if isinstance(df["tokens"].iloc[0], str):
        df["tokens"] = df["tokens"].apply(ast.literal_eval)

    # (선택) 채널명이 있으면 채널 단위로 합쳐서 1채널=1문서로 만들기
    if "channel" in df.columns:
        df = df.groupby("channel", as_index=False)["tokens"].sum()

    # TF-IDF + KMeans
    df = build_clusters(df, num_clusters=5, max_features=1000)

    # C-TF-IDF 키워드
    c_tfidf_matrix, feature_names = compute_c_tfidf(df, max_features=1000)
    top_keywords = extract_top_keywords(c_tfidf_matrix, feature_names, n_top_keywords=10)

    for idx, kws in enumerate(top_keywords):
        print(f"Cluster {idx}: {', '.join(kws)}")

    # LDA
    lda_model, corpus, dictionary, grouped_tokens = train_lda(df, num_topics=5, passes=10)

    # 클러스터(가상 문서) 기준 토픽 확률/대표토픽
    topics_df = get_doc_topics(lda_model, corpus, num_topics=5)

    # 최종 토픽(대표 토픽 최빈값) + 키워드 확인용
    final_topic = topics_df["dominant_topic"].value_counts().idxmax()
    final_keywords = [w for w, _ in lda_model.show_topic(final_topic, topn=10)]
    print(f"\n최종 대표 토픽: Topic {final_topic}")
    print("대표 키워드:", ", ".join(final_keywords))

    # 저장
    save_results(lda_model, num_topics=5, topics_df=topics_df)

    print("모델링 완료!")


if __name__ == "__main__":
    main()
