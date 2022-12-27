from django.urls import path

from . import views

urlpatterns = [
    path('upload', views.upload, name='upload'),
    path('result', views.result, name='result'),
    path('history', views.history, name='history'),
    path('changeModel', views.changeModel, name='changeModel'),
    path('changeModelSubmit', views.changeModelSubmit, name='changeModelSumbit'),
]