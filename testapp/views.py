from django.shortcuts import render
from django.core.paginator import Paginator
from django.db.models import Q
from .models import SteamSearcher
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import re
from datetime import datetime
import pandas as pd
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import time
import os
from pathlib import Path
from django.utils.html import escape
from supabase import create_client, Client
import environ
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
import faiss

# 환경 변수 파일 로드
env = environ.Env()
env_file = os.path.join(Path(__file__).resolve().parent.parent, '.env')
env.read_env(env_file)

# Supabase 설정
supabase_url = env('SUPABASE_URL')
supabase_key = env('SUPABASE_KEY')
supabase: Client = create_client(supabase_url, supabase_key)
supabase.timeout = 30  # 타임아웃을 30초로 설정

# Stopwords 다운로드
nltk.download('punkt')
nltk.download('stopwords')

# GloVe 모델 파일 경로
model_dir = os.path.join(Path(__file__).resolve().parent.parent, "testapp/models")
glove_model_path = os.path.join(model_dir, 'glove.6B.100d.txt')

# GloVe 모델 캐싱
glove_model = {}
stop_words = set(stopwords.words('english'))

def load_glove_model(glove_file_path):
    global glove_model
    if not glove_model:
        print("Loading GloVe model...")
        with open(glove_file_path, 'r', encoding='utf8') as f:
            for line in f:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array([float(val) for val in split_line[1:]])
                glove_model[word] = embedding
        print("GloVe model loaded.")
    return glove_model

load_glove_model(glove_model_path)  # 미리 GloVe 모델 로드

# 텍스트 전처리 함수
def preprocess_text(text):
    words = [word.lower() for word in word_tokenize(text) if word.isalnum() and word.lower() not in stop_words]
    return words

# 텍스트 벡터화 함수
def get_sentence_vector(sentence, embedding_dim=100):
    words = preprocess_text(sentence)
    word_vectors = [glove_model[word] for word in words if word in glove_model]
    if not word_vectors:
        return np.zeros(embedding_dim)
    return np.mean(word_vectors, axis=0)

# Word2Vec 기반 검색어와 유사한 단어를 포함하는 게임을 찾는 함수
def search_games(query, model, embeddings_df):
    query_words = preprocess_text(query)
    if not query_words:
        print("검색어가 너무 짧습니다.")
        return []

    similar_words = []
    for word in query_words:
        if word in model.wv:
            similar_words.extend([w for w, _ in model.wv.most_similar(word, topn=10)])
        else:
            similar_words.append(word)  # 검색어 자체를 유사 단어로 추가

    similar_words = set(similar_words)

    results = []
    seen_game_titles = set()
    for _, row in embeddings_df.iterrows():
        game_title = row['name']
        if game_title in seen_game_titles:
            continue

        game_words = set(row['embedding_words'])
        common_words = game_words.intersection(similar_words)
        if common_words:
            results.append({
                'appid': row['appid'],
                'name': game_title,
                'genre': row['genre'],
                'recommendation_count': row['recommendation_count'],
                'keyphrase': row.get('keyphrase', 'N/A'),
                'summary': row.get('summary', 'N/A'),
            })
            seen_game_titles.add(game_title)

    results.sort(key=lambda x: -x['recommendation_count'])

    return results[:5]

# GloVe 기반 유사도 계산 함수
def find_most_similar_rows(query, embeddings_df, top_n=10):
    query_vector = get_sentence_vector(query)
    embeddings = np.array(embeddings_df['calculated_embedding'].tolist())

    similarities = cosine_similarity([query_vector], embeddings)[0]

    embeddings_df['similarity'] = similarities
    most_similar_rows = embeddings_df.nlargest(top_n, 'similarity')

    results = []
    for _, row in most_similar_rows.iterrows():
        results.append({
            'appid': row['appid'],
            'name': row['name'],
            'genre': row['genre'],
            'recommendation_count': row['recommendation_count'],
            'similarity': row['similarity']
        })

    return results

# FAISS 인덱스 생성
faiss_index = None

def build_faiss_index(embeddings_df):
    global faiss_index
    if faiss_index is None:
        dimension = embeddings_df['calculated_embedding'].iloc[0].size
        faiss_index = faiss.IndexFlatL2(dimension)
        embeddings = np.array(embeddings_df['calculated_embedding'].tolist())
        faiss_index.add(embeddings)
        print(f"FAISS 인덱스가 {dimension} 차원으로 생성되었습니다.")
    return faiss_index

def steam_searcher_list(request):
    search_query = request.GET.get('q', '')

    if not search_query:
        return render(request, 'steam_searcher_list.html', {'page_obj': None, 'search_query': search_query})

    start_time = time.time()

    # SteamSearcher 모델에서 검색
    games = SteamSearcher.objects.filter(
        Q(name__icontains=search_query) |
        Q(genre__icontains=search_query)
    ).order_by('name')

    search_time = time.time()
    print(f"DB 검색 소요 시간: {search_time - start_time:.2f}초")

    model_path = os.path.join(model_dir, 'word2vec_model.bin')
    embed_path = os.path.join(model_dir, 'game_embeddings.pkl')
    saved_embeddings_path = os.path.join(model_dir, 'saved_embeddings.pkl')

    model = Word2Vec.load(model_path)
    embeddings_df = pd.read_pickle(embed_path)
    precomputed_embeddings_df = pd.read_pickle(saved_embeddings_path)

    load_time = time.time()
    print(f"모델 로드 소요 시간: {load_time - search_time:.2f}초")

    # FAISS 인덱스 생성
    build_faiss_index(precomputed_embeddings_df)

    faiss_time = time.time()
    print(f"FAISS 인덱스 생성 소요 시간: {faiss_time - load_time:.2f}초")

    with ThreadPoolExecutor(max_workers=4) as executor:
        word2vec_future = executor.submit(search_games, search_query, model, embeddings_df)
        glove_future = executor.submit(find_most_similar_rows, search_query, precomputed_embeddings_df)

    top_games = word2vec_future.result()
    print(f"Word2Vec 검색 결과: {top_games}")

    glove_results = glove_future.result()
    print(f"GloVe 검색 결과: {glove_results}")

    end_time = time.time()
    print(f"GloVe 검색 소요 시간: {end_time - faiss_time:.2f}초")
    print(f"총 검색 소요 시간: {end_time - start_time:.2f}초")

    paginator = Paginator(games, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    return render(request, 'steam_searcher_list.html', {
        'page_obj': page_obj,
        'search_query': search_query,
        'top_games': top_games,
        'glove_results': glove_results
    })

# 카테고리별로 리뷰를 가져오는 함수
def get_reviews_by_category(category):
    try:
        response = supabase.table('reviews_with_labels').select('*').eq('labels', category).limit(5).execute()
        reviews = response.data
        return reviews
    except Exception as e:
        print(f"Supabase API 오류: {str(e)}")
        return []

# 카테고리별 리뷰 페이지
def category_reviews(request):
    try:
        categories = {
            'gameplay_reviews': get_reviews_by_category('Gameplay'),
            'performance_reviews': get_reviews_by_category('Performance'),
            'visual_and_audio_reviews': get_reviews_by_category('Visual and Audio'),
            'updates_reviews': get_reviews_by_category('Updates'),
            'meaningless_reviews': get_reviews_by_category('Meaningless')
        }
        return render(request, 'category_reviews.html', categories)
    except Exception as e:
        return render(request, 'category_reviews.html', {'error': str(e)})

# 게임 상세 페이지
def game_detail(request, appid):
    search_query = request.GET.get('q', '')
    try:
        game = SteamSearcher.objects.get(appid=appid)
    except SteamSearcher.DoesNotExist:
        game = None
    return render(request, 'game_detail.html', {'game': game, 'search_query': search_query})

# 채팅봇 응답 함수
context = {}

def get_greeting():
    current_hour = datetime.now().hour
    if current_hour < 12:
        return f"Commander Nice to meet you. Now time is {current_hour} o'clock. Good morning."
    elif current_hour < 18:
        return f"Commander Nice to meet you. Now time is {current_hour} o'clock. Good afternoon."
    else:
        return f"Commander Nice to meet you. Now time is {current_hour} o'clock. Good evening."

def get_game_info(game_name):
    try:
        game_name_lower = game_name.lower()

        exact_response = supabase.table('steamsearcher_duplicate').select('appid', 'name').or_(
            f'name.eq.{game_name},keyphrase.eq.{game_name},summary.eq.{game_name}'
        ).execute()
        exact_games = exact_response.data if exact_response.data else []

        partial_response = supabase.table('steamsearcher_duplicate').select('appid', 'name',
                                                                            'recommendation_count').or_(
            f'name.ilike.%{game_name_lower}%,keyphrase.ilike.%{game_name_lower}%,summary.ilike.%{game_name_lower}%'
        ).execute()
        partial_games = partial_response.data if partial_response.data else []

        games = exact_games + [game for game in partial_games if game not in exact_games]

        if games:
            for game in games:
                if game.get('recommendation_count') is None:
                    game['recommendation_count'] = 0

            games = sorted(games, key=lambda x: x['recommendation_count'], reverse=True)

            game_names = [
                f"{i + 1}. <a href='/game/{escape(game['appid'])}/'>{escape(game['name'])}</a> (Good: {escape(str(game['recommendation_count']))})"
                for i, game in enumerate(games[:5])
            ]
            return "Highly recommended games:<br>" + "<br>".join(game_names)
        else:
            return f"Sorry, {escape(game_name)} couldn't find any information about the game."
    except Exception as e:
        return f"Supabase API 오류: {str(e)}"

@csrf_exempt
def chatbot_respond(request):
    if request.method == 'POST':
        user_input = request.POST.get('message', '').lower()

        if re.search("what's your name|what is your name", user_input):
            return JsonResponse({'reply': "I'm Jarvis"}, safe=False)
        if re.search("hello", user_input):
            return JsonResponse({'reply': get_greeting()}, safe=False)
        if re.search("how old are you", user_input):
            return JsonResponse({'reply': "I don't age"}, safe=False)
        if re.search("how are you", user_input):
            return JsonResponse({'reply': "Yes, I'm doing well.."}, safe=False)

        if match_result := re.search("my name is (.*)", user_input):
            name = match_result.group(1)
            context["name"] = name
            return JsonResponse({'reply': f"Nice to meet you {name}. How may I help you?"}, safe=False)

        if match_result := re.search(".*", user_input):
            game_name = match_result.group(0)
            return JsonResponse({'reply': get_game_info(game_name)}, safe=False)

        if "name" in context:
            return JsonResponse({'reply': f"{context['name']}, how may I help you?"}, safe=False)

        if re.search('시간', user_input):
            return JsonResponse({'reply': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, safe=False)

        return JsonResponse({'reply': "Sorry, I didn't quite understand."}, safe=False)

    return JsonResponse({'error': 'Invalid request'}, status=400, safe=False)

