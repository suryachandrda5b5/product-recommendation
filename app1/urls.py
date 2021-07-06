from django.urls import path
from app1 import views
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

urlpatterns = [
    path('hackElite',views.index,name='index.html'),
    path('first',views.firstPage,name='first'),
    path('contact',views.contact,name='contact'),
    path('about',views.about,name='about'),
]
urlpatterns += staticfiles_urlpatterns()