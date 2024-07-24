from django.shortcuts import render
from django.core.paginator import Paginator
from django.db.models import Value, FloatField, Case, When
from .models import SteamSearcher

# 기존 검색
def steam_searcher_list(request):
    search_query = request.GET.get('q', '')

    if not search_query:
        return render(request, 'steam_searcher_list.html', {'page_obj': None, 'search_query': search_query})

    games = SteamSearcher.objects.filter(name__icontains=search_query)

    games = games.annotate(
        recommendation_count_fixed=Case(
            When(recommendation_count__isnull=True, then=Value(0.0)),
            default='recommendation_count',
            output_field=FloatField()
        )
    ).order_by('-recommendation_count_fixed')

    paginator = Paginator(games, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    return render(request, 'steam_searcher_list.html', {'page_obj': page_obj, 'search_query': search_query})

# import os
# import pandas as pd
# from supabase import create_client, Client
# from django.shortcuts import render
# from django.core.paginator import Paginator
# from sentence_transformers import SentenceTransformer
# from konlpy.tag import Okt
# import re
# import torch
# import numpy as np
# from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
#
# # Supabase 설정 (환경 변수에서 URL과 Key를 가져옵니다)
# url = os.getenv("https://nhcmippskpgkykwsumqp.supabase.co")
# key = os.getenv("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5oY21pcHBza3Bna3lrd3N1bXFwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjE2MjYyNzEsImV4cCI6MjAzNzIwMjI3MX0.quApu8EwzqcTgcxdWezDvpZIHSX9LKVQ_NytpLBeAiY")
# supabase: Client = create_client(url, key)
#
# # Supabase에서 데이터를 가져오는 함수
# def fetch_data_from_supabase():
#     response = supabase.table('steam_game_data').select('*').eq('type', 'game').execute()
#     data = response.data
#     df = pd.DataFrame(data)
#     return df
#
# # 불용어 목록 설정 (ENGLISH_STOP_WORDS를 사용)
# stop_words = set(ENGLISH_STOP_WORDS)
#
# # Okt 객체 생성 (한국어 텍스트 처리를 위해 사용)
# okt = Okt()
#
# # 텍스트 전처리 및 구 추출 함수
# def preprocess_and_extract_phrases(text):
#     if text is None:
#         return []
#     # 특수문자 제거
#     text = re.sub(r'[^\w\s]', '', text)
#     # 소문자 변환 및 단어 분리
#     words = text.lower().split()
#     # 불용어 제거
#     words = [word for word in words if word not in stop_words]
#     # 정리된 텍스트
#     cleaned_text = ' '.join(words)
#     # 구 추출
#     phrases = okt.phrases(cleaned_text)
#     return phrases
#
# # 모델 로드 및 GPU 사용 설정
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)
#
# # 각 행의 임베딩을 계산하는 함수
# def calculate_embeddings(row):
#     phrases = row['extracted_phrases']
#     keywords = row['keywords'].split(',')
#     all_texts = phrases + keywords
#     if all_texts:
#         embeddings = model.encode(all_texts, convert_to_tensor=True, device=device)
#         return embeddings.cpu().numpy()  # numpy 배열로 변환
#     else:
#         return np.array([])
#
# # 데이터를 처리하고 저장하는 함수
# def process_and_store_data():
#     # Supabase에서 데이터 가져오기
#     df = fetch_data_from_supabase()
#     # 빈 값 처리
#     df['detailed_description'] = df['detailed_description'].fillna('')
#     df['keywords'] = df['keywords'].fillna('')
#     # 구 추출
#     df['extracted_phrases'] = df['detailed_description'].apply(preprocess_and_extract_phrases)
#     # 임베딩 계산
#     df['embeddings'] = df.apply(calculate_embeddings, axis=1)
#     # 데이터를 파일로 저장
#     df.to_pickle('preprocessed_games.pkl')
#     return df
#
# # 데이터 처리 및 저장 실행
# df = process_and_store_data()
#
# # 검색 기능 구현
# def steam_searcher_list(request):
#     search_query = request.GET.get('q', '')
#
#     if not search_query:
#         return render(request, 'steam_searcher_list.html', {'page_obj': None, 'search_query': search_query})
#
#     # 검색어 임베딩
#     query_embedding = model.encode(search_query, convert_to_tensor=True, device=device).cpu().numpy()
#
#     # 코사인 유사도로 검색
#     def cosine_similarity(a, b):
#         return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
#
#     df['similarity'] = df['embeddings'].apply(lambda x: cosine_similarity(x, query_embedding))
#     results = df.sort_values(by='similarity', ascending=False)
#
#     paginator = Paginator(results.to_dict('records'), 10)
#     page_number = request.GET.get('page')
#     page_obj = paginator.get_page(page_number)
#
#     return render(request, 'steam_searcher_list.html', {'page_obj': page_obj, 'search_query': search_query})