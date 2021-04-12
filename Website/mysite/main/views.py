from django.shortcuts import render
from django.http import HttpResponse
from .models import pred, day, data, day2
from datetime import date
import datetime
from django.db.models import Sum
from django.http import JsonResponse

# Create your views here.

def home(response):
    return render(response, "main/home.html", {})

def Data(response):
    latestPutData = data.objects.all().order_by('-id')[0]
    if str(date.today()) != latestPutData.end:
        newPutData = data(name=f"Putnam County Data {str(date.today())}")
        newPutData.save()
        file = open("../../dataScraper/output.csv", "r")
        for x, line in enumerate(file):
            case = ""
            if x == 2:
                for y, character in enumerate(line):
                    if y >= 24:
                        if character != ",":
                            case = case + character
                        else:
                            day = datetime.date.today() - datetime.timedelta(days=y)
                            newPutData.day2_set.create(date=str(day), cases=int(case))
                            case = ""
    return render(response, "main/data.html", {})

def population_chart(response):
    labels = []
    dataSet = []

    queryset = data.objects.values("data__date")
    for entry in queryset:
        labels.append(entry['data__date'])
        dataSet.append(entry['data__cases'])
    
    return JsonResponse(data={
        'labels': labels,
        'data': dataSet,
    })

def prediction(response):
    latestPutPred = pred.objects.all().order_by('-id')[0]
    if str(date.today()) != latestPutPred.start:
        newPutPred = pred(name=f"Putnam County Pred {str(date.today())}")
        newPutPred.save()
        file = open("../../AI/putPrediction.txt", "r")
        for x, line in enumerate(file):
            day = datetime.date.today() + datetime.timedelta(days=x)
            newPutPred.day_set.create(date=str(day), cases=line)
    return render(response, "main/pred.html", {})

def about(response):
    return render(response, "main/about.html", {})
