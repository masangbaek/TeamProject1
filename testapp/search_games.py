import os
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec

# Stopwords 다운로드
nltk.download('punkt')
nltk.download('stopwords')


# 텍스트 전처리 함수
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = []
    if isinstance(text, list):
        for phrase in text:
            words.extend(
                [word.lower() for word in nltk.word_tokenize(phrase) if word.isalnum() and word not in stop_words])
    else:
        words = [word.lower() for word in nltk.word_tokenize(text) if word.isalnum() and word not in stop_words]
    return words


# 검색어와 유사한 단어를 포함하는 게임을 찾는 함수
def search_games(query, model, embeddings_df):
    query_words = preprocess_text(query)
    if not query_words:
        print("검색어가 너무 짧습니다.")
        return []

    similar_words = []
    for word in query_words:
        if word in model.wv:
            similar_words.extend([w for w, _ in model.wv.most_similar(word, topn=10)])

    similar_words = set(similar_words)  # 유사한 단어들을 집합으로 만듭니다.

    # 유사한 단어를 포함하는 게임 찾기
    results = []
    for _, row in embeddings_df.iterrows():
        game_words = set(row['embedding_words'])
        common_words = game_words.intersection(similar_words)
        if common_words:
            results.append({
                'name': row['name'],
                'genre': row['genre'],
                'recommendation_count': row['recommendation_count'],
                # 2024-07-26
                'keyphrase': row.get('keyphrase', 'N/A'),  # 필드 추가
                'summary': row.get('summary', 'N/A'),  # 필드 추가
                'common_words': common_words
            })

    results.sort(key=lambda x: -x['recommendation_count'])

    return results[:5]


# 검색어 입력 및 결과 출력
def main_search():
    query = input("검색어를 입력하세요: ")

    # 모델 로드 경로 설정
    model_dir = "testapp/models"
    model_path = os.path.join(model_dir, 'word2vec_model.bin')
    embed_path = os.path.join(model_dir, 'game_embeddings.pkl')

    model = Word2Vec.load(model_path)
    embeddings_df = pd.read_pickle(embed_path)

    top_games = search_games(query, model, embeddings_df)

    print(f"검색어: {query}")
    for game in top_games:
        print(f"Name: {game['name']}, Genre: {game['genre']}, Recommendation Count: {game['recommendation_count']}, {game['keyphrase']}, {game['summary']}")
        print(f"Common Words: {', '.join(game['common_words'])}")


# 검색 함수 호출
if __name__ == "__main__":
    main_search()
