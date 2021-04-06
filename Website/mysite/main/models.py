from django.db import models
from datetime import date

# Create your models here.

class pred(models.Model):
    name = models.CharField(max_length=50)
    start = models.CharField(max_length=10, default=str(date.today()))

    def __str__(self):
        return self.name

class data(models.Model):
    name = models.CharField(max_length=50)
    end = models.CharField(max_length=10, default=str(date.today()))

    def __str__(self):
        return self.name

class day(models.Model):
    pred = models.ForeignKey(pred, on_delete=models.CASCADE)
    date = models.CharField(max_length=10)
    cases = models.IntegerField()

    def __str__(self):
        return self.date