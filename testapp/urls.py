# 기존 설정
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path, include
from .views import steam_searcher_list, game_detail, chatbot_respond, category_reviews

urlpatterns = [
    path('', steam_searcher_list, name='home'),
    path('game/<int:appid>/', game_detail, name='game_detail'),
    # 챗봇 응답 URL 패턴 추가
    path('chatbot/respond/', chatbot_respond, name='chatbot_respond'),

    path('categories/', category_reviews, name='category_reviews'),  # 새로운 경로 추가


]
# 추가: 정적 파일 및 미디어 파일 서빙
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

