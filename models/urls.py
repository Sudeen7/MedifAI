from django.urls import path
from . import views

urlpatterns = [
    path('', views.landingpage, name='landing'),
    path('medifai/', views.index, name='medifai'),
    path('predict/', views.predict, name='predict'),
    path('about/', views.about, name='about'),  
    path('contact/', views.contact_us, name='contact'),
    path('thank-you/', views.thank_you, name='thank_you'),  
    path('developer/', views.developer, name='developer'),  
    path('blog/', views.blog, name='blog'),  
    path('privacy/', views.privacy, name='privacy'),  
    path('terms/', views.terms, name='terms'),  
    path('signup/', views.SignupPage, name='signup'),
    path('login/', views.LoginPage, name='login'),
    path('home/', views.HomePage, name='homepage'),
    path('logout/',views.LogoutPage,name='logout'),
    path('diabetes-predict/', views.predict_diabetes, name='diabetes_predict'),
    path('heart-disease-predict/', views.heart_disease_predict, name='heart_disease_predict'),
]


