from django.shortcuts import render
from django.core.paginator import Paginator
from django.db.models import Q, Value, FloatField, Case, When
from .models import SteamSearcher

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

    return render(request, 'steam_searcher_list.html', {'page_obj': page_obj, 'search_query': search_query})

# 게임 상세 페이지
def game_detail(request, appid):
    search_query = request.GET.get('q', '')
    game = SteamSearcher.objects.get(appid=appid)
    return render(request, 'game_detail.html', {'game': game, 'search_query': search_query})
