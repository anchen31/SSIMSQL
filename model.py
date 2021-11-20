import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
import math
# from collections import deque
# import random
# import time
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
# from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
# from sqlalchemy import create_engine
# import pymysql
# pymysql.install_as_MySQLdb()
# import mysql.connector
# from mysql.connector import Error

import config

password = config.password

min_max_scaler = preprocessing.MinMaxScaler()

def scaleColumns(df, cols_to_scale):
    for col in cols_to_scale:
        df[col] = pd.DataFrame(min_max_scaler.fit_transform(pd.DataFrame(df[col])),columns=[col])
    return df

def isSupport(df,i):
  support = df['low'][i] < df['low'][i-1]  and df['low'][i] < df['low'][i+1] and df['low'][i+1] < df['low'][i+2] and df['low'][i-1] < df['low'][i-2]
  return support
def isResistance(df,i):
  resistance = df['high'][i] > df['high'][i-1]  and df['high'][i] > df['high'][i+1] and df['high'][i+1] > df['high'][i+2] and df['high'][i-1] > df['high'][i-2]
  return resistance

################################################################## pull the data and read it 

# try:
#     con = mysql.connector.connect(
#     host = 'localhost',
#     database='twitterdb', 
#     user='root', 
#     password = password)

#     cursor = con.cursor()
#     # create data db with full data set
#     query = "select * from ibpy"
#     cursor.execute(query)
#     # get all records
#     db = cursor.fetchall()

#     df = pd.DataFrame(db)

# except mysql.connector.Error as e:
#     print("Error reading data from MySQL table", e)

#     cursor.close()
#     con.close()

# df = df.set_axis(['date', 'open', 'high', 
#                     'low', 'close', 'volume', 
#                     'average', 'barCount', 'bb_bbm', 
#                     'bb_bbh', 'bb_bbl', 'VWAP', 
#                     'RSI', 'STsupp', 'STres', 
#                     'LTsupp', 'LTres', 'GLD', 
#                     'UVXY', 'SQQQ'], axis=1, inplace=False)


df = pd.read_csv('data.csv')

# stores support and resistance into data
data = []

for i in range(2,df.shape[0]-2):
  if isSupport(df,i):
    data.append((i ,df['low'][i], 1))
  elif isResistance(df,i):
    data.append((i ,df['high'][i], -1))


# normalizes data and also labels data from 0-1
cols = list(df)[2:]
df_for_training = df[cols].astype(float)
scaled_df = scaleColumns(df_for_training,['open', 'high', 
                    'low', 'close', 'volume', 
                    'average', 'barCount', 'bb_bbm', 
                    'bb_bbh', 'bb_bbl', 'VWAP', 
                    'RSI', 'STsupp', 'STres', 
                    'LTsupp', 'LTres', 'GLD', 
                    'UVXY', 'SQQQ'])

# plots the buy/sell classifcation from the s/r data
scaled_df['trade'] = 0.5
for stuff in data:
	if stuff[2] == 1:
		scaled_df['trade'][stuff[0]] = 1
	if stuff[2] == -1:
		scaled_df['trade'][stuff[0]] = 0


# Create a new dataframe with only the 'Adj close column'
new_data = scaled_df.filter(['trade'])


# Convert the dataframe to a numpy array
dataset = scaled_df.to_numpy()
# Get the number of rows to train the model on
len_train = math.ceil(len(dataset)*.67)

train = dataset[0:len_train, :].reshape(-1,1)
test = dataset[len_train:len(dataset), :1].reshape(-1,1)

print(train)


# 

X_train, y_train = [], []
for i in range(len(train)-80-1):
    a = train[i:(i+80), 0]
    X_train.append(a)
    y_train.append(train[i+80, 0])
    X_train = np.array(X_train)
    y_train = np.array(y_train)


# print(scaled_df.columns)


# scaled_df.plot()
# plt.show()



plt.plot(train)
plt.show()