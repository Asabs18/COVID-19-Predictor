from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("testChart", views.population_chart, name="population-chart"),
    path("COVID-19-Data", views.Data, name="data"),
    path("COVID-19-Predictions", views.prediction, name="prediction"),
    path("About-Us", views.about, name="about"),
]