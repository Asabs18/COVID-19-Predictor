import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

csv_path = "../dataScraper/output.csv"

import csv
      
x = []
y = []

df = pd.read_csv(csv_path)
df = df.drop(['countyFIPS'], 1)
df = df.drop(['County Name'], 1)
df = df.drop(['State'], 1)
df = df.drop(['StateFIPS'], 1)

n = df.shape[0]
p = df.shape[1]

df = df.values


for val in df[0]:
    print(val)

n = np.size(df)
train_start = 0
train_end = int(np.floor(0.8*n))
test_start = train_end
test_end = n
data_train = df[np.arange(train_start, train_end), :]
data_test = df[np.arange(test_start, test_end), :]

print(data_train)
print(data_test)

print(df[np.arange(train_start, train_end), :])

# for label, content in df.items():
#     print(f'label: {label} content: {content}')
#     if label[0] == "2":
#         x.append(label)
#         y.append(int(content))

# plt.figure(figsize=(17,5))
# plt.plot(x,y, label='Number of Cases')
# plt.xlabel('Time')
# plt.ylabel('Cases')
# plt.title('COVID NUMBERS PUTNAM COUNTY')
# plt.legend()
# plt.show()


# df = pd.read_csv(csv_path)
# print(df)