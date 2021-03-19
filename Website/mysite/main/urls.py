from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("COVID-19-Data", views.data, name="data"),
    path("COVID-19-Predictions", views.prediction, name="prediction"),
    path("About-Us", views.about, name="about"),
]