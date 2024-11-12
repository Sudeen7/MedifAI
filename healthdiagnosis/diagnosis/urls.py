from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.predict, name='predict'),
    path('about/', views.about, name='about'),  
    path('contact/', views.contact, name='contact'),  
    path('developer/', views.developer, name='developer'),  
    path('blog/', views.blog, name='blog'),  
    path('privacy/', views.privacy, name='privacy'),  
    path('terms/', views.terms, name='terms'),  
]


