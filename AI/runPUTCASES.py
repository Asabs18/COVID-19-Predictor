import math, os, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from keras.models import Sequential
from keras.layers import Dense, LSTM

'''
A LSTM Machine learning model that takes current covid data from Putnam County NY 
and passses that through a neural network to predict future COVID-19 Cases.
'''

#Get the Data from the scraper
csv_path = "../dataScraper/output.csv"

#Read in the data
df = pd.read_csv(csv_path)

#get rid of not needed values
df = df.drop(['countyFIPS'], 1)
df = df.drop(['County Name'], 1)
df = df.drop(['State'], 1)
df = df.drop(['StateFIPS'], 1)

#invert the rows and columns and set the column title
df = df.transpose()
df.columns = ['Cases']

#convert to numpy array
data = df.filter(['Cases'])
dataset = data.values

#get the length of the training data 
training_data_len = math.ceil(len(dataset))

#scale the data
scaler = QuantileTransformer(output_distribution='uniform')
scaled_data = scaler.fit_transform(dataset)

#Build the LSTM (Loading in pretrained model)
model = keras.models.load_model('models/PUTCASES/model')

#Get the Test data
finalPreds = []
for i in range(60):
    x_test = []
    x_test.append(scaled_data[-60:])
    #format test data
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    #make prediction
    predictions = model.predict(x_test)
    scaled_data = np.append(scaled_data, predictions[0])
    scaled_data = scaled_data.transpose()
    predictions = scaler.inverse_transform(predictions)
    finalPreds.append(predictions)

for i, val in enumerate(finalPreds):
    print(f"Day {i+1}: {math.ceil(val[0])} Cases predicted")