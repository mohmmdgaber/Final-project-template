from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('Home', views.index, name='index'),
    path('register', views.reg, name='register'),
    path('regsubmit', views.regsubmit, name='regsubmit'),
    path('login', views.loginpage, name='login'),
    path('loginsubmit', views.loginsubmit, name='loginsubmit'),
    path('userHome', views.userHome, name='userHome'),
    path('adminHome', views.adminHome, name='adminHome'),
    path('viewUsers', views.viewUsers, name='viewUsers'),
    path('deleteUser/<username>/', views.deleteUser, name='deleteUser'),
]