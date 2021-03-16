import math, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import QuantileTransformer
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
training_data_len = math.ceil(len(dataset) * .8)

#scale the data (CHANGE FROM A MINMAXSCALER TO A SCALER WITH NO CAP FOR BETTER RESULTS)
scaler = QuantileTransformer(output_distribution='uniform')
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

#Build the LSTM (Loading in pretrained model)
#model = keras.models.load_model('models/PUTCASES/model')

model = Sequential()
model.add(LSTM(50, return_sequences = True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer="adam", loss='mean_squared_error')

#Train the Model
model.fit(x_train, y_train, batch_size=1, epochs=3)

#Get the Test data
test_data = scaled_data[training_data_len - 60: , :]
x_test = []
y_test = dataset[training_data_len: , :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

#format test data
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#make prediction
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


#get the average error
rmse = np.sqrt( np.mean( predictions - y_test )**2 )

#Plot the results
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title('COVID-19 Predicted Cases(Training)')
plt.xticks(np.arange(0, 1000, 50))
plt.xlabel('Date', fontsize=8)
plt.ylabel('Number of Cases', fontsize=18)
plt.plot(train['Cases'])
plt.plot(valid[['Cases', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
print(rmse)
if input("Would you like to save this model and overwrite the previous model? (y/n)").lower() == 'y':
    model.save('models/PUTCASES/model')

#OPTIMIZERS
    #class Adam: Optimizer that implements the Adam algorithm.
    #class Nadam: Optimizer that implements the NAdam algorithm.
    #https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
    #This ^^^ website has info on all of the listed optimizers
    #https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adamax
    #^^^Adamax optimizer^^^

#RETRAIN CODE:
    #model = Sequential()
    #model.add(LSTM(50, return_sequences = True, input_shape=(x_train.shape[1], 1)))
    #model.add(LSTM(50, return_sequences = False))
    #model.add(Dense(25))
    #model.add(Dense(1))    
    #model.compile(optimizer="rmsprop", loss='mean_squared_error')

#Train the Model
    #model.fit(x_train, y_train, batch_size=1, epochs=3)

#MAYBE  OPTIMIZERS  :
    #class Adamax: Optimizer that implements the Adamax algorith
    #class RMSprop: Optimizer that implements the RMSprop algorithm.
    #class SGD: Gradient descent (with momentum) optimizer.

#SCALERS:
    #QuantileTransformer(output_distribution='normal')

#MAYBE SCALERS:
    #MaxAbsScaler()
    #RobustScaler(quantile_range=(25, 75))
    #PowerTransformer(method='yeo-johnson')