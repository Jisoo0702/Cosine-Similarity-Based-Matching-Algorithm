# 관계인구 활성화를 위한 인플루언서 매칭 시스템

인구감소지역의 관계인구 활성화를 목표로,  
관광지 데이터와 인플루언서 콘텐츠 데이터를 분석하여  
지역 특성에 적합한 인플루언서를 매칭하는 추천 시스템이다.

관광지 데이터는 한국관광공사가 운영하는 공식 관광 플랫폼  
**‘대한민국 구석구석’** 사이트를 크롤링하여 수집하였으며,  
인플루언서 데이터는 Playboard 및 YouTube Data API를 활용해 구축하였다.

---

## Pipeline Overview

관광지 데이터  
→ 텍스트 전처리  
→ LDA  
→ 지역별 핵심 키워드 추출  
→ FastText 임베딩  

인플루언서 데이터  
→ 텍스트 전처리  
→ TF-IDF  
→ 방문 목적 키워드 추출  
→ FastText 임베딩  

지역 키워드 벡터 × 인플루언서 키워드 벡터  
→ Cosine Similarity  
→ 지역별 인플루언서 매칭

---

## Key Features
- 공공 관광 데이터 및 유튜브 콘텐츠 기반 데이터 수집
- LDA, TF-IDF를 활용한 지역·콘텐츠 키워드 추출
- FastText 임베딩을 통한 의미적 유사성 반영
- Cosine Similarity 기반 지역–인플루언서 매칭

---

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Gensim (LDA, FastText)

---

## Directory Structure

```plaintext
project/
├─ data/
│  ├─ tourism_data.csv
│  └─ influencer_data.csv
├─ preprocessing/
│  └─ text_preprocessing.py
├─ modeling/
│  ├─ lda_model.ipynb
│  ├─ tfidf_analysis.ipynb
│  └─ fasttext_embedding.ipynb
├─ matching/
│  ├─ cosine_similarity.py
│  └─ influencer_matching.ipynb
└─ README.md
