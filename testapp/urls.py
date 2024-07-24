from django.urls import path
from .views import steam_searcher_list

urlpatterns = [
    path('steamsearcher/', steam_searcher_list, name='steam_searcher_list'),
]
