from pathlib import Path
import ast

import fasttext
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


TOURISM_PATH = Path("data/matching/tourism_keywords.xlsx")
INFLUENCER_PATH = Path("data/matching/influencer_keywords.xlsx")
FASTTEXT_BIN_PATH = Path("models/cc.ko.300.bin")

OUTPUT_PATH = Path("data/matching/recommendation_results_fasttext.xlsx")


# 키워드 파싱
def parse_keywords(x) -> list[str]:
    if pd.isna(x):
        return []
    x = str(x).strip()

    # "['a','b']" 같은 리스트 문자열이면 복원
    if x.startswith("[") and x.endswith("]"):
        try:
            return [w.strip() for w in ast.literal_eval(x) if str(w).strip()]
        except:
            pass

    # "a, b, c" 형태면 split
    return [w.strip() for w in x.split(",") if w.strip()]


# 단어 벡터 안전 추출
def safe_get_vector(word: str, model) -> np.ndarray | None:
    try:
        return model.get_word_vector(word)
    except:
        return None


# 키워드 리스트 평균 벡터
def mean_vector(keywords: list[str], model) -> np.ndarray | None:
    vectors = []
    for w in keywords:
        v = safe_get_vector(w, model)
        if v is not None:
            vectors.append(v)

    return np.mean(vectors, axis=0) if vectors else None


# 채널별 Top-N 키워드 (방문_횟수 가중치 기반)
def build_weighted_top_keywords(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    def calc(group: pd.DataFrame) -> list[str]:
        counts = {}
        for _, row in group.iterrows():
            weight = row.get("방문_횟수", 1)
            for kw in row["키워드"]:
                counts[kw] = counts.get(kw, 0) + weight

        return [k for k, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]]

    grouped = df.groupby("채널명").apply(calc).reset_index()
    grouped.columns = ["채널명", "Top_Keywords"]
    return grouped


# 코사인 유사도 기반 매칭
def match_by_cosine(
    tourism_df: pd.DataFrame,
    influencer_raw_df: pd.DataFrame,
    influencer_grouped_df: pd.DataFrame,
    threshold: float = 0.1,
) -> pd.DataFrame:
    results = []

    for _, region_row in tourism_df.iterrows():
        region_vector = region_row["Vector"]
        if region_vector is None:
            continue

        for _, ch_row in influencer_grouped_df.iterrows():
            ch_vector = ch_row["Vector"]
            if ch_vector is None:
                continue

            sim = cosine_similarity([region_vector], [ch_vector])[0][0]
            if sim < threshold:
                continue

            channel_name = ch_row["채널명"]
            channel_data = influencer_raw_df[influencer_raw_df["채널명"] == channel_name]

            avg_views = channel_data["평균_조회수"].mean() if "평균_조회수" in channel_data.columns else None
            avg_likes = channel_data["평균_영상_좋아요"].mean() if "평균_영상_좋아요" in channel_data.columns else None
            subs = channel_data["구독자 수"].iloc[0] if "구독자 수" in channel_data.columns else None
            target = channel_data["여행 대상"].iloc[0] if "여행 대상" in channel_data.columns else None

            results.append(
                {
                    "권역": region_row.get("권역"),
                    "지역": region_row.get("지역"),
                    "채널명": channel_name,
                    "코사인 유사도": round(float(sim), 3),
                    "지역 키워드": ", ".join(region_row["관광지 키워드"]),
                    "채널 키워드": ", ".join(ch_row["Top_Keywords"]),
                    "평균 조회수": avg_views,
                    "평균 좋아요 수": avg_likes,
                    "구독자 수": subs,
                    "여행 대상": target,
                }
            )

    return pd.DataFrame(results)


# 실행
def main() -> None:
    tourism_df = pd.read_excel(TOURISM_PATH)
    influencer_df = pd.read_excel(INFLUENCER_PATH)

    model = fasttext.load_model(str(FASTTEXT_BIN_PATH))

    # 키워드 컬럼 파싱
    tourism_df["관광지 키워드"] = tourism_df["관광지 키워드"].apply(parse_keywords)
    influencer_df["키워드"] = influencer_df["키워드"].apply(parse_keywords)

    # 채널별 Top-Keywords 생성 (가중치 방식)
    influencer_grouped = build_weighted_top_keywords(influencer_df, top_n=20)

    # Top-Keywords 20개 미만 채널 제거
    missing = influencer_grouped[influencer_grouped["Top_Keywords"].apply(len) < 20]["채널명"].tolist()
    if missing:
        print("상위 20개 키워드 부족 채널:")
        print("\n".join(missing))

    influencer_grouped = influencer_grouped[influencer_grouped["Top_Keywords"].apply(len) == 20]

    # 벡터 생성
    tourism_df["Vector"] = tourism_df["관광지 키워드"].apply(lambda x: mean_vector(x, model))
    influencer_grouped["Vector"] = influencer_grouped["Top_Keywords"].apply(lambda x: mean_vector(x, model))

    # 매칭
    rec_df = match_by_cosine(
        tourism_df=tourism_df,
        influencer_raw_df=influencer_df,
        influencer_grouped_df=influencer_grouped,
        threshold=0.1,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rec_df.to_excel(OUTPUT_PATH, index=False, engine="openpyxl")

    print(f"매칭 결과 저장 완료: {OUTPUT_PATH}")
    print(f"총 추천 결과: {len(rec_df)}")


if __name__ == "__main__":
    main()
