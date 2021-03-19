from django.shortcuts import render
from django.http import HttpResponse
from .models import pred, day
from datetime import date
import datetime

# Create your views here.

def home(response):
    latestPutPred = pred.objects.all().order_by('-id')[0]
    if str(date.today()) != latestPutPred.start:
        newPutPred = pred(name=f"Putnam County Pred {str(date.today())}")
        newPutPred.save()
        file = open("../../AI/putPrediction.txt", "r")
        for x, line in enumerate(file):
            day = datetime.date.today() + datetime.timedelta(days=x)
            newPutPred.day_set.create(date=str(day), cases=line)
    return render(response, "main/home.html", {})

def data(response):
    return render(response, "main/data.html", {})

def prediction(response):
    return render(response, "main/pred.html", {})

def about(response):
    return render(response, "main/about.html", {})
