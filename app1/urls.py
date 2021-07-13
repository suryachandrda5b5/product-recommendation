from django.urls import path
from app1 import views
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

urlpatterns = [
    path('codeassassins',views.index,name='index.html'),
    path('first',views.firstPage,name='first'),
    path('contact',views.contact,name='contact'),
    path('about',views.about,name='about'),
    path('recommend',views.recommend,name='recommend'),
]
urlpatterns += staticfiles_urlpatterns()
