# 2025-07-24
from django.shortcuts import render
from django.core.paginator import Paginator
from django.db.models import Q, Value, FloatField, Case, When
from .models import SteamSearcher
# 챗봇 기능 2024-07-25
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import re
from datetime import datetime
from supabase import create_client, Client
#스팀 기사 호출하는 함수 추가 2024-07-25
import feedparser

# def get_game_news(request):
#     try:
#         params = {
#             'q': 'game OR gaming OR video game',  # 검색어
#             'language': 'en',  # 언어 설정
#             'sortBy': 'publishedAt',  # 최신 뉴스 기준 정렬
#             'apiKey': NEWS_API_KEY  # API 키
#         }
#         response = requests.get(NEWS_API_URL, params=params)
#         response.raise_for_status()  # 오류가 발생하면 예외 발생
#
#         news_data = response.json()
#         articles = news_data.get('articles', [])
#         return JsonResponse({'articles': articles[:5]})  # 최신 기사 5개만 반환
#     except Exception as e:
#         print(f"Error fetching game news: {str(e)}")
#         return JsonResponse({'error': 'Error fetching news'}, status=500)


# 수정 내용
def steam_searcher_list(request):
    search_query = request.GET.get('q', '')

    if not search_query:
        return render(request, 'steam_searcher_list.html', {'page_obj': None, 'search_query': search_query})

    # 기존 games = SteamSearcher.objects.filter(name__icontains=search_query)
    # name, keyphrase, summary 필드에서 검색
    games = SteamSearcher.objects.filter(
            Q(name__icontains=search_query) |
            Q(keyphrase__icontains=search_query) |
            Q(summary__icontains=search_query)

        )

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

    return render(request,
                  'steam_searcher_list.html', {
                   'page_obj': page_obj,
                   'search_query': search_query,
    })


# 게임 상세 페이지
def game_detail(request, appid):
    search_query = request.GET.get('q', '')
    game = SteamSearcher.objects.get(appid=appid)
    return render(request, 'game_detail.html', {'game': game, 'search_query': search_query})

# 2024-07-25
# Supabase 설정 (챗봇을 위한 설정)
supabase_url = "https://nhcmippskpgkykwsumqp.supabase.co"
supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5oY21pcHBza3Bna3lrd3N1bXFwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjE2MjYyNzEsImV4cCI6MjAzNzIwMjI3MX0.quApu8EwzqcTgcxdWezDvpZIHSX9LKVQ_NytpLBeAiY"
supabase: Client = create_client(supabase_url, supabase_key)

context = {}

def get_greeting():
    current_hour = datetime.now().hour
    if current_hour < 12:
        return "지휘관님 반갑습니다. 현재 시각은 {}시로 좋은 아침입니다.".format(current_hour)
    elif current_hour < 18:
        return "지휘관님 반갑습니다. 현재 시각은 {}시로 좋은 오후입니다.".format(current_hour)
    else:
        return "지휘관님 반갑습니다. 현재 시각은 {}시로 즐거운 저녁시간을 보내세요.".format(current_hour)


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
        partial_response = supabase.table('steamsearcher_duplicate').select('appid', 'name', 'recommendation_count').or_(
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
            return f"죄송하지만, 게임 {game_name}에 대한 정보를 찾을 수 없습니다."
    except Exception as e:
        return f"Supabase API 오류: {str(e)}"


# 챗봇 응답을 처리하는 뷰
@csrf_exempt
def chatbot_respond(request):
    global context  # 전역 변수 context 사용

    if request.method == 'POST':
        user_input = request.POST.get('message').lower()  # 사용자의 입력을 소문자로 변환

        # 입력된 메시지에 따라 다른 응답 반환
        if re.search("이름이 뭐야", user_input):
            return JsonResponse({'reply': "저는 자비스에요"}, safe=False)
        if re.search("안녕", user_input):
            return JsonResponse({'reply': get_greeting()}, safe=False)
        if re.search("몇 살이야", user_input):
            return JsonResponse({'reply': "저는 나이를 먹지 않아요"}, safe=False)
        if re.search("잘 지내?", user_input):
            return JsonResponse({'reply': "네 잘 지내고 있습니다."}, safe=False)

        # 사용자가 자신의 이름을 입력한 경우
        match_result = re.search("내 이름은 (.*)", user_input)
        if match_result:
            name = match_result.group(1)
            context["name"] = name  # 이름을 context에 저장
            return JsonResponse({'reply': f"{name}님 반갑습니다. 무엇을 도와드릴까요?"}, safe=False)

        # 게임 정보를 요청한 경우
        match_result = re.search("게임 (.*) 정보", user_input)
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
        return JsonResponse({'reply': "죄송하지만 잘 이해하지 못했습니다."}, safe=False)

    # 잘못된 요청인 경우
    return JsonResponse({'error': 'Invalid request'}, status=400, safe=False)
