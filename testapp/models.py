from django.db import models

# 기존 검색
class SteamSearcher(models.Model):
    appid = models.BigIntegerField(primary_key=True)
    name = models.CharField(max_length=255)
    genre = models.CharField(max_length=255)
    detailed_description = models.TextField()
    recommendation_count = models.FloatField()
    type = models.CharField(max_length=255)
    developer = models.CharField(max_length=255)
    release_date = models.TextField()  # 날짜 형식이 특정되지 않았으므로 TextField로 사용
    keyphrase = models.CharField(max_length=255) # 추가
    summary = models.CharField(max_length=1024) # 추가


    def __str__(self):
        return self.name

    class Meta:
        db_table = 'steamsearcher_duplicate'  # 실제 테이블 이름으로 설정
