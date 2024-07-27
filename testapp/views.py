from django.shortcuts import render
from django.core.paginator import Paginator
from django.db.models import Q, Value, FloatField, Case, When
from .models import SteamSearcher
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import re
from datetime import datetime
from supabase import create_client, Client
import pandas as pd
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
import time
import os

# Supabase 설정
supabase_url = "https://nhcmippskpgkykwsumqp.supabase.co"
supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5oY21pcHBza3Bna3lrd3N1bXFwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjE2MjYyNzEsImV4cCI6MjAzNzIwMjI3MX0.quApu8EwzqcTgcxdWezDvpZIHSX9LKVQ_NytpLBeAiY"
supabase: Client = create_client(supabase_url, supabase_key)

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


# 기존 코드
def steam_searcher_list(request):
    search_query = request.GET.get('q', '')

    if not search_query:
        return render(request, 'steam_searcher_list.html', {'page_obj': None, 'search_query': search_query})

    games = SteamSearcher.objects.filter(
        Q(name__icontains=search_query) |
        Q(keyphrase__icontains=search_query) |
        Q(summary__icontains=search_query),
        ~Q(genre=None),
        ~Q(description_phrases=None)
    )

    games = games.annotate(
        recommendation_count_fixed=Case(
            When(recommendation_count__isnull=True, then=Value(0.0)),
            default='recommendation_count',
            output_field=FloatField()
        )
    ).order_by('-recommendation_count_fixed')

    # description_phrases를 문자열로 변환하여 템플릿에 전달합니다.
    for game in games:
        if isinstance(game.description_phrases, list):
            game.description_phrases = ', '.join(game.description_phrases)

    # 모델 로드 경로 설정
    model_dir = "testapp/models"
    model_path = os.path.join(model_dir, 'word2vec_model.bin')
    embed_path = os.path.join(model_dir, 'game_embeddings.pkl')

    model = Word2Vec.load(model_path)
    embeddings_df = pd.read_pickle(embed_path)

    # 검색어와 유사한 게임 찾기
    start_time = time.time()  # 타이머 시작
    top_games = search_games(search_query, model, embeddings_df)
    end_time = time.time()  # 타이머 종료

    print(f"검색어와 유사한 게임 찾기 소요 시간: {end_time - start_time}초")
    print(f"검색된 게임 수: {len(top_games)}")  # 추가된 디버깅 메시지

    paginator = Paginator(games, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    return render(request,
                  'steam_searcher_list.html', {
                      'page_obj': page_obj,
                      'search_query': search_query,
                      'top_games': top_games  # 템플릿에 top_games 전달
                  })


# 게임 상세 페이지
def game_detail(request, appid):
    search_query = request.GET.get('q', '')
    game = SteamSearcher.objects.get(appid=appid)
    return render(request, 'game_detail.html', {'game': game, 'search_query': search_query})


def get_greeting():
    current_hour = datetime.now().hour
    if current_hour < 12:
        return "Commander Nice to meet you. Now time is {}o'clock Good morning.".format(current_hour)
    elif current_hour < 18:
        return "Commander Nice to meet you. Now time is {}o'clock Good afternoon.".format(current_hour)
    else:
        return "Commander Nice to meet you. Now time is {}o'clock Good evening.".format(current_hour)


def get_game_info(game_name):
    try:
        # 검색어를 소문자로 변환
        game_name_lower = game_name.lower()

        # 정확히 일치하는 게임을 먼저 검색
        exact_response = supabase.table('steamsearcher_duplicate').select('appid', 'name', 'recommendation_count').or_(
            f'name.eq.{game_name},keyphrase.eq.{game_name},summary.eq.{game_name}'
        ).execute()
        # 검색 결과가 있으면 데이터를 가져오고, 없으면 빈 리스트를 반환
        exact_games = exact_response.data if exact_response.data else []

        # 부분 일치 검색
        partial_response = supabase.table('steamsearcher_duplicate').select('appid', 'name',
                                                                            'recommendation_count').or_(
            f'name.ilike.%{game_name_lower}%,keyphrase.ilike.%{game_name_lower}%,summary.ilike.%{game_name_lower}%'
        ).execute()
        # 검색 결과가 있으면 데이터를 가져오고, 없으면 빈 리스트를 반환
        partial_games = partial_response.data if partial_response.data else []

        # 정확히 일치하는 게임을 우선 순위로 설정
        games = exact_games + [game for game in partial_games if game not in exact_games]

        if games:
            # 추천 수가 None인 경우 0으로 변환
            for game in games:
                if game['recommendation_count'] is None:
                    game['recommendation_count'] = 0

            # 추천 수로 정렬
            games = sorted(games, key=lambda x: x['recommendation_count'], reverse=True)

            # 게임 목록을 링크 포함하여 생성
            game_names = [
                f"{i + 1}. <a href='/game/{game['appid']}/'>{game['name']}</a> (추천 수: {game['recommendation_count']})"
                for i, game in enumerate(games[:5])
            ]
            # HTML 형식으로 반환
            return "추천 수가 높은 게임:<br>" + "<br>".join(game_names)
        else:
            return f"Sorry, {game_name} couldn't find any information about game "
    except Exception as e:
        return f"Supabase API 오류: {str(e)}"


# 챗봇 응답을 처리하는 뷰
@csrf_exempt
def chatbot_respond(request):
    global context  # 전역 변수 context 사용

    if request.method == 'POST':
        user_input = request.POST.get('message').lower()  # 사용자의 입력을 소문자로 변환

        # 입력된 메시지에 따라 다른 응답 반환
        if re.search("What your name?", user_input):
            return JsonResponse({'reply': "I'm Jarvis"}, safe=False)
        if re.search("Hello", user_input):
            return JsonResponse({'reply': get_greeting()}, safe=False)
        if re.search("How old are you?", user_input):
            return JsonResponse({'reply': "I don't age"}, safe=False)
        if re.search("How are you?", user_input):
            return JsonResponse({'reply': "Yes, I'm doing well.."}, safe=False)

        # 사용자가 자신의 이름을 입력한 경우
        match_result = re.search("My name is (.*)", user_input)
        if match_result:
            name = match_result.group(1)
            context["name"] = name  # 이름을 context에 저장
            return JsonResponse({'reply': f" Nice to meet you {name} How may I help you?"}, safe=False)

        # 게임 정보를 요청한 경우
        match_result = re.search("(.*)", user_input)
        if match_result:
            game_name = match_result.group(1)
            return JsonResponse({'reply': get_game_info(game_name)}, safe=False)

        # context에 이름이 저장되어 있는 경우
        if "name" in context:
            return JsonResponse({'reply': f"{context['name']}님 무엇을 도와드릴까요?"}, safe=False)

        # 현재 시간을 요청한 경우
        if re.search('시간', user_input):
            return JsonResponse({'reply': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, safe=False)

        # 이해할 수 없는 입력인 경우
        return JsonResponse({'reply': "Sorry, I didn't quite understand."}, safe=False)

    # 잘못된 요청인 경우
    return JsonResponse({'error': 'Invalid request'}, status=400, safe=False)

