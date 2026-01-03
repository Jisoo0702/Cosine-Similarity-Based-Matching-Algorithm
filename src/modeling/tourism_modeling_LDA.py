from pathlib import Path
import ast

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from gensim import corpora
from gensim.models import LdaModel


TOKENS_DATA_PATH = Path("data/tourism_boryeong_tokens.csv")  # tourism_preprocess_tokens.py 결과
OUTPUT_TOPICS_PATH = Path("data/modeling/tourism_topics.csv")
OUTPUT_DOC_TOPICS_PATH = Path("data/modeling/tourism_doc_topics.csv")


# TF-IDF + 클러스터링
def build_clusters(df: pd.DataFrame, num_clusters: int = 5, max_features: int = 1000) -> pd.DataFrame:
    df = df.copy()
    df["joined_tokens"] = df["tokens"].apply(lambda x: " ".join(x))

    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(df["joined_tokens"])

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(tfidf_matrix)

    return df


# C-TF-IDF
def compute_c_tfidf(df: pd.DataFrame, num_clusters: int, max_features: int = 1000):
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
        top_keywords.append([feature_names[i] for i in top_indices])

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
def save_results(
    lda_model: LdaModel,
    num_topics: int,
    topics_df: pd.DataFrame,
):
    OUTPUT_TOPICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 토픽별 키워드 저장
    topic_rows = []
    for topic_id in range(num_topics):
        words = [w for w, _ in lda_model.show_topic(topic_id, topn=10)]
        topic_rows.append(
            {
                "topic_id": topic_id,
                "keywords": ", ".join(words),
            }
        )

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

    # TF-IDF + KMeans
    df = build_clusters(df, num_clusters=5, max_features=1000)

    # C-TF-IDF 키워드
    c_tfidf_matrix, feature_names = compute_c_tfidf(df, num_clusters=5, max_features=1000)
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
