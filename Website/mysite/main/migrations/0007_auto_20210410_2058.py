# Generated by Django 3.1.4 on 2021-04-11 00:58

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0006_day2'),
    ]

    operations = [
        migrations.AlterField(
            model_name='data',
            name='end',
            field=models.CharField(default='2021-04-10', max_length=10),
        ),
        migrations.AlterField(
            model_name='day2',
            name='cases',
            field=models.IntegerField(default=0),
        ),
        migrations.AlterField(
            model_name='day2',
            name='date',
            field=models.CharField(default='2021-04-10', max_length=10),
        ),
        migrations.AlterField(
            model_name='pred',
            name='start',
            field=models.CharField(default='2021-04-10', max_length=10),
        ),
    ]
