import os, datetime, math, csv
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

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
training_data_len = math.ceil(len(dataset) * .8)

#scale the data (CHANGE FROM A MINMAXSCALER TO A SCALER WITH NO CAP FOR BETTER RESULTS)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#Create the Training Dataset
#The timesteps is how far back in the data you look, may need to be adjusted
timesteps = 60
train_data = scaled_data[0:training_data_len, :]
x_train = []
y_train = []
for i in range(timesteps, len(train_data)):
    x_train.append(train_data[i-timesteps:i, 0])
    y_train.append(train_data[i, 0])
    
#Convert the x and y train dataset to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the Data
x_train = np.reshape(x_train, (x_train.shape[0], timesteps, 1))

print(f"our training dataset has the shape of {x_train.shape}")
print("X Train:")
print(x_train)
print("Y Train:")
print(x_train)
#Build the LSTM