# from django.shortcuts import render
# from django.core.paginator import Paginator
# from django.db.models import Q, Value, FloatField, Case, When
# from .models import SteamSearcher
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# import re
# from datetime import datetime
# import pandas as pd
# from gensim.models import Word2Vec
# import nltk
# from nltk.corpus import stopwords
# import time
# import os
# from pathlib import Path
# from django.utils.html import escape
# from supabase import create_client, Client
# import environ
#
# # 환경 변수 파일 로드
# env = environ.Env()
# env_file = os.path.join(Path(__file__).resolve().parent.parent, '.env')
# env.read_env(env_file)
#
# # Supabase 설정
# supabase_url = env('SUPABASE_URL')
# supabase_key = env('SUPABASE_KEY')
# supabase: Client = create_client(supabase_url, supabase_key)
#
# # Stopwords 다운로드
# nltk.download('punkt')
# nltk.download('stopwords')
#
# # 텍스트 전처리 함수
# def preprocess_text(text):
#     stop_words = set(stopwords.words('english'))
#     words = []
#     if isinstance(text, list):
#         for phrase in text:
#             words.extend(
#                 [word.lower() for word in nltk.word_tokenize(phrase) if word.isalnum() and word not in stop_words])
#     else:
#         words = [word.lower() for word in nltk.word_tokenize(text) if word.isalnum() and word not in stop_words]
#     return words
#
# # 검색어와 유사한 단어를 포함하는 게임을 찾는 함수
# def search_games(query, model, embeddings_df):
#     query_words = preprocess_text(query)
#     if not query_words:
#         print("검색어가 너무 짧습니다.")
#         return []
#
#     similar_words = []
#     for word in query_words:
#         if word in model.wv:
#             similar_words.extend([w for w, _ in model.wv.most_similar(word, topn=10)])
#
#     similar_words = set(similar_words)
#
#     results = []
#     seen_game_titles = set()
#     for _, row in embeddings_df.iterrows():
#         game_title = row['name']
#         if game_title in seen_game_titles:
#             continue
#
#         game_words = set(row['embedding_words'])
#         common_words = game_words.intersection(similar_words)
#         if common_words:
#             results.append({
#                 'name': game_title,
#                 'genre': row['genre'],
#                 'recommendation_count': row['recommendation_count'],
#                 'keyphrase': row.get('keyphrase', 'N/A'),
#                 'summary': row.get('summary', 'N/A'),
#             })
#             seen_game_titles.add(game_title)
#
#     results.sort(key=lambda x: -x['recommendation_count'])
#
#     return results[:5]
#
# # 카테고리별로 리뷰를 가져오는 함수
# def get_reviews_by_category(category):
#     response = supabase.table('reviews_with_labels').select('*').eq('labels', category).limit(5).execute()
#     reviews = response.data
#     return reviews
#
# def steam_searcher_list(request):
#     search_query = request.GET.get('q', '')
#
#     if not search_query:
#         return render(request, 'steam_searcher_list.html', {'page_obj': None, 'search_query': search_query})
#
#     games = SteamSearcher.objects.filter(
#         Q(name__icontains=search_query) |
#         Q(keyphrase__icontains=search_query) |
#         Q(summary__icontains=search_query),
#         ~Q(detailed_description=search_query) |
#         ~Q(genre=search_query),
#     )
#
#     # for game in games:
#     #     if isinstance(game.description_phrases, list):
#     #         game.description_phrases = ', '.join(game.description_phrases)
#
#     model_dir = "testapp/models"
#     model_path = os.path.join(model_dir, 'word2vec_model.bin')
#     embed_path = os.path.join(model_dir, 'game_embeddings.pkl')
#
#     model = Word2Vec.load(model_path)
#     embeddings_df = pd.read_pickle(embed_path)
#
#     start_time = time.time()
#     top_games = search_games(search_query, model, embeddings_df)
#     end_time = time.time()
#
#     print(f"검색어와 유사한 게임 찾기 소요 시간: {end_time - start_time}초")
#     print(f"검색된 게임 수: {len(top_games)}")
#
#     paginator = Paginator(games, 10)
#     page_number = request.GET.get('page')
#     page_obj = paginator.get_page(page_number)
#
#     return render(request, 'steam_searcher_list.html', {'page_obj': page_obj, 'search_query': search_query, 'top_games': top_games})
#
# # 카테고리별 리뷰 페이지
# def category_reviews(request):
#     categories = {
#         'Gameplay': get_reviews_by_category('Gameplay'),
#         'Performance': get_reviews_by_category('Performance'),
#         'Visual and Audio': get_reviews_by_category('Visual and Audio'),
#         'Updates': get_reviews_by_category('Updates'),
#         'Meaningless': get_reviews_by_category('Meaningless')
#     }
#     return render(request, 'category_reviews.html', {'categories': categories})
#
# # 게임 상세 페이지
# def game_detail(request, appid):
#     search_query = request.GET.get('q', '')
#     game = SteamSearcher.objects.get(appid=appid)
#
#     return render(request, 'game_detail.html', {'game': game, 'search_query': search_query})
#
# def get_greeting():
#     current_hour = datetime.now().hour
#     if current_hour < 12:
#         return "Commander Nice to meet you. Now time is {}o'clock Good morning.".format(current_hour)
#     elif current_hour < 18:
#         return "Commander Nice to meet you. Now time is {}o'clock Good afternoon.".format(current_hour)
#     else:
#         return "Commander Nice to meet you. Now time is {}o'clock Good evening.".format(current_hour)
#
# def get_game_info(game_name):
#     try:
#         game_name_lower = game_name.lower()
#
#         exact_response = supabase.table('steamsearcher_duplicate').select('appid', 'name').or_(
#             f'name.eq.{game_name},keyphrase.eq.{game_name},summary.eq.{game_name}'
#         ).execute()
#         exact_games = exact_response.data if exact_response.data else []
#
#         partial_response = supabase.table('steamsearcher_duplicate').select('appid', 'name',
#                                                                             'recommendation_count').or_(
#             f'name.ilike.%{game_name_lower}%,keyphrase.ilike.%{game_name_lower}%,summary.ilike.%{game_name_lower}%'
#         ).execute()
#         partial_games = partial_response.data if partial_response.data else []
#
#         games = exact_games + [game for game in partial_games if game not in exact_games]
#
#         if games:
#             for game in games:
#                 if game.get('recommendation_count') is None:
#                     game['recommendation_count'] = 0
#
#             games = sorted(games, key=lambda x: x['recommendation_count'], reverse=True)
#
#             game_names = [
#                 f"{i + 1}. <a href='/game/{escape(game['appid'])}/'>{escape(game['name'])}</a> (Good: {escape(str(game['recommendation_count']))})"
#                 for i, game in enumerate(games[:5])
#             ]
#             return "Highly recommended games:<br>" + "<br>".join(game_names)
#         else:
#             return f"Sorry, {escape(game_name)} couldn't find any information about game "
#     except Exception as e:
#         return f"Supabase API 오류: {str(e)}"
#
# @csrf_exempt
# def chatbot_respond(request):
#     global context
#
#     if request.method == 'POST':
#         user_input = request.POST.get('message').lower()
#
#         if re.search("What your name?", user_input):
#             return JsonResponse({'reply': "I'm Jarvis"}, safe=False)
#         if re.search("Hello", user_input):
#             return JsonResponse({'reply': get_greeting()}, safe=False)
#         if re.search("How old are you?", user_input):
#             return JsonResponse({'reply': "I don't age"}, safe=False)
#         if re.search("How are you?", user_input):
#             return JsonResponse({'reply': "Yes, I'm doing well.."}, safe=False)
#
#         if match_result := re.search("My name is (.*)", user_input):
#             name = match_result.group(1)
#             context["name"] = name
#             return JsonResponse({'reply': f" Nice to meet you {name} How may I help you?"}, safe=False)
#
#         if match_result := re.search("(.*)", user_input):
#             game_name = match_result.group(1)
#             return JsonResponse({'reply': get_game_info(game_name)}, safe=False)
#
#         if "name" in context:
#             return JsonResponse({'reply': f"{context['name']} How may I help you?"}, safe=False)
#
#         if re.search('시간', user_input):
#             return JsonResponse({'reply': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, safe=False)
#
#         return JsonResponse({'reply': "Sorry, I didn't quite understand."}, safe=False)
#
#     return JsonResponse({'error': 'Invalid request'}, status=400, safe=False)


from django.shortcuts import render
from django.core.paginator import Paginator
from django.db.models import Q, Value, FloatField, Case, When
from .models import SteamSearcher
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import re
from datetime import datetime
import pandas as pd
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
import time
import os
from pathlib import Path
from django.utils.html import escape
from supabase import create_client, Client
import environ

# 환경 변수 파일 로드
env = environ.Env()
env_file = os.path.join(Path(__file__).resolve().parent.parent, '.env')
env.read_env(env_file)

# Supabase 설정
supabase_url = env('SUPABASE_URL')
supabase_key = env('SUPABASE_KEY')
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
                'name': game_title,
                'genre': row['genre'],
                'recommendation_count': row['recommendation_count'],
                'keyphrase': row.get('keyphrase', 'N/A'),
                'summary': row.get('summary', 'N/A'),
            })
            seen_game_titles.add(game_title)

    results.sort(key=lambda x: -x['recommendation_count'])

    return results[:5]

# 카테고리별로 리뷰를 가져오는 함수
def get_reviews_by_category(category):
    response = supabase.table('reviews_with_labels').select('*').eq('labels', category).limit(5).execute()
    reviews = response.data
    return reviews

def steam_searcher_list(request):
    search_query = request.GET.get('q', '')

    if not search_query:
        return render(request, 'steam_searcher_list.html', {'page_obj': None, 'search_query': search_query})

    games = SteamSearcher.objects.filter(
        Q(name__icontains=search_query) |
        Q(keyphrase__icontains=search_query) |
        Q(summary__icontains=search_query),
        ~Q(detailed_description=search_query) |
        ~Q(genre=search_query),
    )

    # description_phrases 참조 부분 주석 처리
    # for game in games:
    #     if isinstance(game.description_phrases, list):
    #         game.description_phrases = ', '.join(game.description_phrases)

    model_dir = "testapp/models"
    model_path = os.path.join(model_dir, 'word2vec_model.bin')
    embed_path = os.path.join(model_dir, 'game_embeddings.pkl')

    model = Word2Vec.load(model_path)
    embeddings_df = pd.read_pickle(embed_path)

    start_time = time.time()
    top_games = search_games(search_query, model, embeddings_df)
    end_time = time.time()

    print(f"검색어와 유사한 게임 찾기 소요 시간: {end_time - start_time}초")
    print(f"검색된 게임 수: {len(top_games)}")

    paginator = Paginator(games, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    return render(request, 'steam_searcher_list.html', {'page_obj': page_obj, 'search_query': search_query, 'top_games': top_games})

# 카테고리별 리뷰 페이지
def category_reviews(request):
    categories = {
        'gameplay_reviews': get_reviews_by_category('Gameplay'),
        'performance_reviews': get_reviews_by_category('Performance'),
        'visual_and_audio_reviews': get_reviews_by_category('Visual and Audio'),
        'updates_reviews': get_reviews_by_category('Updates'),
        'meaningless_reviews': get_reviews_by_category('Meaningless')
    }
    return render(request, 'category_reviews.html', categories)

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
            return f"Sorry, {escape(game_name)} couldn't find any information about game "
    except Exception as e:
        return f"Supabase API 오류: {str(e)}"

@csrf_exempt
def chatbot_respond(request):
    global context

    if request.method == 'POST':
        user_input = request.POST.get('message').lower()

        if re.search("What your name?", user_input):
            return JsonResponse({'reply': "I'm Jarvis"}, safe=False)
        if re.search("Hello", user_input):
            return JsonResponse({'reply': get_greeting()}, safe=False)
        if re.search("How old are you?", user_input):
            return JsonResponse({'reply': "I don't age"}, safe=False)
        if re.search("How are you?", user_input):
            return JsonResponse({'reply': "Yes, I'm doing well.."}, safe=False)

        if match_result := re.search("My name is (.*)", user_input):
            name = match_result.group(1)
            context["name"] = name
            return JsonResponse({'reply': f" Nice to meet you {name} How may I help you?"}, safe=False)

        if match_result := re.search("(.*)", user_input):
            game_name = match_result.group(1)
            return JsonResponse({'reply': get_game_info(game_name)}, safe=False)

        if "name" in context:
            return JsonResponse({'reply': f"{context['name']} How may I help you?"}, safe=False)

        if re.search('시간', user_input):
            return JsonResponse({'reply': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, safe=False)

        return JsonResponse({'reply': "Sorry, I didn't quite understand."}, safe=False)

    return JsonResponse({'error': 'Invalid request'}, status=400, safe=False)
