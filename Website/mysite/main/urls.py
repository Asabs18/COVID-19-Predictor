from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("testChartD", views.population_chartD, name="population-chartD"),
    path("COVID-19-Data", views.Data, name="data"),
    path("testChartP", views.population_chartP, name="population-chartP"),
    path("COVID-19-Predictions", views.prediction, name="prediction"),
    path("About-Us", views.about, name="about"),
]