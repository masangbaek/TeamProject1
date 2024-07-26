import os
import time
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from supabase import create_client, Client

# Stopwords 다운로드
nltk.download('punkt')
nltk.download('stopwords')

# Supabase 클라이언트 설정
url = "https://nhcmippskpgkykwsumqp.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5oY21pcHBza3Bna3lrd3N1bXFwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjE2MjYyNzEsImV4cCI6MjAzNzIwMjI3MX0.quApu8EwzqcTgcxdWezDvpZIHSX9LKVQ_NytpLBeAiY"
supabase: Client = create_client(url, key)


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


# Word2Vec 모델 훈련 및 저장 함수
def train_and_save_word2vec(games, model_path):
    sentences = [preprocess_text(game['description_phrases']) for game in games if game['description_phrases']]
    sentences = [sentence for sentence in sentences if sentence]  # 빈 문장 제거
    print(f"전처리된 문장 수: {len(sentences)}")  # 디버깅용 출력
    start_time = time.time()
    model = Word2Vec(sentences, vector_size=150, window=7, min_count=2, workers=4, epochs=10, seed=42)
    end_time = time.time()
    print(f"훈련 시간: {end_time - start_time:.2f}초")
    model.save(model_path)
    print(f"Word2Vec 모델이 {model_path}에 저장되었습니다.")


# 페이지네이션을 사용하여 모든 데이터를 가져오는 함수
def fetch_all_games():
    limit = 1000
    offset = 0
    all_games = []

    while True:
        print(f"Fetching games with offset: {offset}")
        response = supabase.table('steamsearcher_duplicate').select(
            'appid, name, genre, recommendation_count, description_phrases', 'keyphrase', 'summary' ).range(offset, offset + limit - 1).execute()
        games = response.data
        if not games:
            break
        all_games.extend(games)
        offset += limit
        print(f"가져온 게임 데이터 개수: {len(all_games)}")  # 진행 상황 출력

    return all_games


# 게임 데이터를 미리 임베딩하고 저장하는 함수
def embed_and_save_game_data(model_path, embed_path):
    games = fetch_all_games()
    print(f"총 가져온 게임 데이터 개수: {len(games)}")

    model = Word2Vec.load(model_path)
    embeddings = []

    for game in games:
        if not game['description_phrases']:
            continue
        game_words = preprocess_text(game['description_phrases'])
        game_vectors = [model.wv[word] for word in game_words if word in model.wv]
        if not game_vectors:
            continue
        game_embedding = np.mean(game_vectors, axis=0)
        embeddings.append({
            'appid': game['appid'],
            'name': game['name'],
            'genre': game['genre'],
            'recommendation_count': game['recommendation_count'],
            # 추가
            'keyphrase': game['keyphrase'],
            'summary': game['summary'],
            'embedding': game_embedding,
            'embedding_words': game_words
        })

    df = pd.DataFrame(embeddings)
    df.to_pickle(embed_path)
    print(f"임베딩 데이터가 {embed_path}에 저장되었습니다.")


# 임베딩 저장 함수 호출
if __name__ == "__main__":
    model_dir = "testapp/models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'word2vec_model.bin')

    embed_path = os.path.join(model_dir, 'game_embeddings.pkl')

    # 모델 학습 및 저장
    games = fetch_all_games()
    print(f"총 가져온 게임 수: {len(games)}")
    train_and_save_word2vec(games, model_path)

    # 게임 데이터 임베딩 및 저장
    embed_and_save_game_data(model_path, embed_path)
