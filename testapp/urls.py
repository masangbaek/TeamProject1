from django.urls import path
from .views import steam_searcher_list, game_detail # game_detail 추가

urlpatterns = [
    path('steamsearcher/', steam_searcher_list, name='steam_searcher_list'),
    path('game/<int:appid>/', game_detail, name='game_detail'), # 추가
]
