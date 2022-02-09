import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
import math
# from collections import deque
import random
import time
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


from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
# from darts.models import TransformerModel
# from darts.metrics import mape, rmse
# from darts.utils.timeseries_generation import datetime_attribute_timeseries
# from darts.utils.likelihood_models import QuantileRegression

import config

password = config.password

min_max_scaler = preprocessing.MinMaxScaler()



LOAD = False         # True = load previously saved model from disk?  False = (re)train the model
SAVE = ""   # file name to save the model under

EPOCHS = 200
INLEN = 32          # input size
FEAT = 32           # d_model = number of expected features in the inputs, up to 512    
HEADS = 4           # default 8
ENCODE = 4          # encoder layers
DECODE = 4          # decoder layers
DIM_FF = 128        # dimensions of the feedforward network, default 2048
BATCH = 32          # batch size
ACTF = "relu"       # activation function, relu (default) or gelu
SCHLEARN = None     # a PyTorch learning rate scheduler; None = constant rate
LEARN = 1e-3        # learning rate
VALWAIT = 1         # epochs to wait before evaluating the loss on the test/validation set
DROPOUT = 0.1       # dropout rate
N_FC = 1            # output size

RAND = 42           # random seed
N_SAMPLES = 100     # number of times a prediction is sampled from a probabilistic model
N_JOBS = 3          # parallel processors to use;  -1 = all processors

# default quantiles for QuantileRegression
QUANTILES = [0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99]

SPLIT = 0.7         # train/test %

FIGSIZE = (9, 6)


qL1, qL2 = 0.01, 0.10        # percentiles of predictions: lower bounds
qU1, qU2 = 1-qL1, 1-qL2,     # upper bounds derived from lower bounds
label_q1 = f'{int(qU1 * 100)} / {int(qL1 * 100)} percentile band'
label_q2 = f'{int(qU2 * 100)} / {int(qL2 * 100)} percentile band'

mpath = os.path.abspath(os.getcwd()) + SAVE     # path and file name to save the model






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


df = pd.read_csv('OPdata.csv')
# df['date'] = pd.to_datetime(df['date'])
# df = df.set_index('date')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')] # removes the unamed df dolumn

# stores support and resistance into data
data = []

for i in range(2,df.shape[0]-2):
  if isSupport(df,i):
    data.append((i ,df['low'][i], 1))
  elif isResistance(df,i):
    data.append((i ,df['high'][i], -1))





# normalizes data and also labels data from 0-1
cols = list(df)[1:]
# print(cols)

# might not need this if the model and scale on its own
df_for_training = df[cols].astype(float)
# print(df_for_training)
scaled_df = scaleColumns(df_for_training,['open', 'high', 
                    'low', 'close', 'volume', 
                    'average', 'barCount', 'bb_bbm', 
                    'bb_bbh', 'bb_bbl', 'VWAP', 
                    'RSI', 'STsupp', 'STres', 
                    'LTsupp', 'LTres', 'GLD', 
                    'UVXY', 'SQQQ'])
# print(df_for_training)


# plots the buy/sell classifcation from the s/r data
df_for_training['trade'] = 0.5
for stuff in data:
	if stuff[2] == 1:
		df_for_training.at[stuff[0], 'trade'] = 1
	if stuff[2] == -1:
		df_for_training.at[stuff[0], 'trade'] = 0

# Convert the dataframe to a numpy array
dataset = df_for_training.to_numpy()

# print(scaled_df.columns)



# create time series object for target variable
ts_P = TimeSeries.from_series(df_for_training["trade"]) 

# creates time series object covariate feature
df_covF = df_for_training.loc[:, df_for_training.columns != "trade"]
ts_covF = TimeSeries.from_dataframe(df_covF)

# train/test split and scaling of target variable
ts_train, ts_test = ts_P.split_after(SPLIT)

scalerP = Scaler()
scalerP.fit_transform(ts_train)
ts_ttrain = scalerP.transform(ts_train)
ts_ttest = scalerP.transform(ts_test)    
ts_t = scalerP.transform(ts_P)

# make sure data are of type float
ts_t = ts_t.astype(np.float32)
ts_ttrain = ts_ttrain.astype(np.float32)
ts_ttest = ts_ttest.astype(np.float32)


# train/test split and scaling of feature covariates
covF_train, covF_test = ts_covF.split_after(SPLIT)

scalerF = Scaler()
scalerF.fit_transform(covF_train)
covF_ttrain = scalerF.transform(covF_train) 
covF_ttest = scalerF.transform(covF_test)   
covF_t = scalerF.transform(ts_covF)  

# make sure data are of type float
covF_t = covF_t.astype(np.float32)
covF_ttrain = covF_ttrain.astype(np.float32)
covF_ttest = covF_ttest.astype(np.float32)














# Get the number of rows to train the model on
len_train = math.ceil(len(dataset)*.67)

# train = dataset[0:len_train, :].reshape(-1,1)
# test = dataset[len_train:len(dataset), :1].reshape(-1,1)


trainX = [] # the amount of samples and n features
trainY = [] # the target shape

n_future = 1   # Number of days we want to look into the future based on the past days.
n_past = 14  # Number of past days we want to use to predict the future.

#Reformat input data into a shape: (n_samples x timesteps x n_features)
#In my example, my df_for_training_scaled has a shape (12823, 5)
#12823 refers to the number of data points and 5 refers to the columns (multi-variables).

for i in range(n_past, len(dataset) - n_future +1):
    # trainX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
    trainX.append(dataset[i - n_past:i, 0:19])
    trainY.append(dataset[i + n_future - 1:i + n_future, 19]) # 19 is the column with the target that I want to test

trainX, trainY = np.array(trainX), np.array(trainY)

# print('trainX shape == {}.'.format(trainX.shape))
# print('trainY shape == {}.'.format(trainY.shape))











# model = Sequential()
# model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
# model.add(LSTM(32, activation='relu', return_sequences=False))
# model.add(Dropout(0.2))
# model.add(Dense(trainY.shape[1]))

# opt = tf.keras.optimizers.Adam(lr=0.01, decay=1e-6)

# model.compile(loss='sparse_categorical_crossentropy',
#                 optimizer=opt,
#                 metrics=['accuracy'])

# model.fit(trainX, trainY, epochs=10, batch_size=32)


# model = Sequential()
# model.add(LSTM(64, input_shape=trainX.shape[1:], return_sequences=True))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())

# model.add(LSTM(64, input_shape=trainX.shape[1:], return_sequences=True))
# model.add(Dropout(0.1))
# model.add(BatchNormalization())

# model.add(LSTM(64, input_shape=trainX.shape[1:]))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())

# model.add(Dense(32, activation="relu"))
# model.add(Dropout(0.2))

# model.add(Dense(2, activation="softmax"))

# opt = tf.keras.optimizers.Adam(lr=0.01, decay=1e-6)

# model.compile(loss='sparse_categorical_crossentropy',
#                 optimizer=opt,
#                 metrics=['accuracy'])

# model.fit(trainX, trainY, epochs=10, batch_size=32)



# model = Sequential()
# #Adding the first LSTM layer and some Dropout regularisation
# model.add(LSTM(units = 50, return_sequences = True, input_shape = (trainX.shape[1], 1)))
# model.add(Dropout(0.2))
# # Adding a second LSTM layer and some Dropout regularisation
# model.add(LSTM(units = 50, return_sequences = True))
# model.add(Dropout(0.2))
# # Adding a third LSTM layer and some Dropout regularisation
# model.add(LSTM(units = 50, return_sequences = True))
# model.add(Dropout(0.2))
# # Adding a fourth LSTM layer and some Dropout regularisation
# model.add(LSTM(units = 50))
# model.add(Dropout(0.2))
# # Adding the output layer
# model.add(Dense(units = 1))

# # Compiling the RNN
# model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# # Fitting the RNN to the Training set
# model.fit(trainX, trainY, epochs = 10, batch_size = 32)





# df_for_training = df_for_training.filter(['open', 'STsupp', 'STres', 
#                     'LTsupp', 'LTres'])
del scaled_df['trade']
scaled_df.plot()
plt.show()

# plt.plot(trainY)
# plt.show()