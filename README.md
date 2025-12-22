# 관계인구 활성화를 위한 인플루언서 매칭 시스템

본 프로젝트는 인구감소지역의 관계인구 활성화를 목표로,  
관광지 데이터와 인플루언서 데이터를 분석하여  
지역 특성에 적합한 인플루언서를 매칭하는 시스템이다.

지역과 인플루언서 간의 의미적 유사도를 반영하기 위해  
텍스트 분석과 코사인 유사도 기반 매칭 알고리즘을 활용한다.

---

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Gensim (LDA, FastText)
- Matplotlib
- Jupyter Notebook

---

## 주요 기능

### 1. 관광지 데이터 분석
- 관광지 설명 텍스트 전처리
- LDA 토픽 모델링을 통해 관광자원 중심의 지역별 핵심 키워드 추출

### 2. 인플루언서 데이터 분석
- 인플루언서 콘텐츠(해시태그, 설명문) 전처리
- TF-IDF를 적용하여 지역별 방문 목적 및 콘텐츠 성향 키워드 추출

### 3. 단어 임베딩
- 지역 키워드 및 인플루언서 키워드를 FastText 기반 벡터로 변환
- 단어 간 의미적 유사성 반영

### 4. 유사도 계산
- 지역 키워드 벡터와 인플루언서 키워드 벡터 간 Cosine Similarity 계산
- 지역–인플루언서 간 유사도 점수 산출

### 5. 인플루언서 매칭
- 유사도 점수가 높은 인플루언서를 지역별로 매칭
- 지역 특성에 적합한 인플루언서 추천 결과 도출

---

## 파이프라인

관광지 데이터  
→ LDA  
→ 지역별 키워드 추출  
→ FastText 임베딩  

인플루언서 데이터  
→ TF-IDF  
→ 방문 목적 키워드 추출  
→ FastText 임베딩  

지역 키워드 벡터 × 인플루언서 키워드 벡터  
→ Cosine Similarity  
→ 인플루언서 매칭

---

## 디렉토리 구조

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
