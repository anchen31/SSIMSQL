import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
import math
# from collections import deque
import random
import time
import seaborn as sns
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
# from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# from sqlalchemy import create_engine
# import pymysql
# pymysql.install_as_MySQLdb()
# import mysql.connector
# from mysql.connector import Error

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TransformerModel
from darts.metrics import mape, rmse
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression

import config

password = config.password

min_max_scaler = preprocessing.MinMaxScaler()
# scaler = MinMaxScaler(feature_range = (0,1))
scaler = StandardScaler()


LOAD = False         # True = load previously saved model from disk?  False = (re)train the model
SAVE = "/_TForm_model10e.pth.tar"   # file name to save the model under

EPOCHS = 40
INLEN = 32          # input size
FEAT = 32           # d_model = number of expected features in the inputs, up to 512    
HEADS = 4           # default 8
ENCODE = 5          # encoder layers
DECODE = 5          # decoder layers
DIM_FF = 64        # dimensions of the feedforward network, default 2048
BATCH = 32          # batch size
ACTF = "relu"       # activation function, relu (default) or gelu
SCHLEARN = None     # a PyTorch learning rate scheduler; None = constant rate
LEARN = 1e-4        # learning rate
VALWAIT = 1         # epochs to wait before evaluating the loss on the test/validation set
DROPOUT = 0.1       # dropout rate
N_FC = 1            # output size

RAND = 42           # random seed
N_SAMPLES = 100     # number of times a prediction is sampled from a probabilistic model
N_JOBS = 3          # parallel processors to use;  -1 = all processors

# default quantiles for QuantileRegression
QUANTILES = [0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99]

SPLIT = 0.9         # train/test %

FIGSIZE = (9, 6)


qL1, qL2, qL3 = 0.01, 0.05, 0.10        # percentiles of predictions: lower bounds
qU1, qU2, qU3 = 1-qL1, 1-qL2, 1-qL3     # upper bounds derived from lower bounds
label_q1 = f'{int(qU1 * 100)} / {int(qL1 * 100)} percentile band'
label_q2 = f'{int(qU2 * 100)} / {int(qL2 * 100)} percentile band'
label_q3 = f'{int(qU3 * 100)} / {int(qL3 * 100)} percentile band'

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
data = []

df1 = pd.read_csv('OPdata.csv')
df1 = df1.loc[:, ~df1.columns.str.contains('^Unnamed')] # removes the unamed df dolumn
del df1['volume']
del df1['barCount']
del df1['RSI']

# gets the local mins and maxes
for i in range(2,df1.shape[0]-2):
  if isSupport(df1, i):
    data.append((i, df1['low'][i], 1))
  elif isResistance(df1,i):
    data.append((i, df1['high'][i], -1))

# df1['trade'] = 5
# for stuff in data:
#   if stuff[2] == 1:
#     df1.iat[stuff[0], 18] = 6
#   if stuff[2] == -1:
#     df1.iat[stuff[0], 18] = 4

# df1['date'] = pd.to_datetime(df1['date'])
# df1 = df1.set_index('date')
ts = df1['open']
# trade = df1['trade']

# df1 = df1.reset_index()
# xaxis = df1['date'].values.tolist()
# x = range(len(df1))


df1 = df1.drop(columns=['date'])



df = df1.copy()

# df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df1.index)

############################################################## create multiple time series object 
fill = False
# create time series object for target variable, This is univariate
ts_P = TimeSeries.from_series(df["open"], fill_missing_dates=fill, freq=None)
ts_P = ts_P.pd_dataframe()
ts_P_1 = ts_P.fillna(method='ffill')
ts_P = TimeSeries.from_series(ts_P_1)

# creates time series object covariate feature, This is multivariate
df_covF = df.loc[:, df.columns != "open"]
ts_covF = TimeSeries.from_dataframe(df_covF, fill_missing_dates=fill, freq=None)
ts_covF = ts_covF.pd_dataframe()
ts_covF_1 = ts_covF.fillna(method='ffill')
ts_covF = TimeSeries.from_series(ts_covF_1)

############################################################## splits data into train or test data

# train/test split and scaling of TARGET variable
ts_train, ts_test = ts_P.split_after(SPLIT)

scalerP = Scaler(scaler)
# scalerP.fit_transform(ts_train)
scalerP.fit_transform(ts_P) # LETS SEE IF THIS WORKSSSSS
ts_ttrain = scalerP.transform(ts_train)
ts_ttest = scalerP.transform(ts_test)    
ts_t = scalerP.transform(ts_P)

# # make sure data are of type float
# ts_t = ts_t.astype(np.float32)
# ts_ttrain = ts_ttrain.astype(np.float32)
# ts_ttest = ts_ttest.astype(np.float32)


# train/test split and scaling of FEATURE covariates
covF_train, covF_test = ts_covF.split_after(SPLIT)

scalerF = Scaler(scaler)
# scalerF.fit_transform(covF_train)
scalerF.fit_transform(ts_covF) # LETS SEE IF THIS WORKSSSSS
covF_ttrain = scalerF.transform(covF_train) 
covF_ttest = scalerF.transform(covF_test)   
covF_t = scalerF.transform(ts_covF)  

# # make sure data are of type float
# covF_ttrain = ts_ttrain.astype(np.float32)
# covF_ttest = ts_ttest.astype(np.float32)


# feature engineering - create time covariates: hour, weekday, month, year, country-specific holidays



#################################################################### graphs the cycles of the data

# cov_t = covF_t.pd_dataframe()
# df3 = cov_t
# print(len(df3))
# df3.plot()
# plt.show()


# plt.figure(figsize = (15,15))
# sns.set(font_scale=0.75)
# ax = sns.heatmap(df3.corr().round(3), 
#             annot=True, 
#             square=True, 
#             linewidths=.75, cmap="coolwarm", 
#             fmt = ".2f", 
#             annot_kws = {"size": 11})
# ax.xaxis.tick_bottom()
# plt.title("correlation matrix")
# plt.show()



###############################################################################################

model = TransformerModel(
                    input_chunk_length = INLEN,
                    output_chunk_length = N_FC,
                    batch_size = BATCH,
                    n_epochs = EPOCHS,
                    model_name = "Transformer_price_noFill",
                    nr_epochs_val_period = VALWAIT,
                    d_model = FEAT,
                    nhead = HEADS,
                    num_encoder_layers = ENCODE,
                    num_decoder_layers = DECODE,
                    dim_feedforward = DIM_FF,
                    dropout = DROPOUT,
                    activation = ACTF,
                    random_state=RAND,
                    likelihood=QuantileRegression(quantiles=QUANTILES), 
                    optimizer_kwargs={'lr': LEARN},
                    save_checkpoints=True,
                    force_reset=True
                    )


# training: load a saved model or (re)train
if LOAD:
    print("have loaded a previously saved model from disk:" + mpath)
    model = TransformerModel.load_model(mpath)                            # load previously model from disk 
else:
    model.fit(  ts_ttrain, 
                past_covariates=covF_t, 
                verbose=True)
    print("have saved the model after training:", mpath)
    model.save_model(mpath)

# # testing: generate predictions
ts_tpred = model.predict(   n=len(ts_ttest), 
                            num_samples=N_SAMPLES,   
                            n_jobs=N_JOBS, 
                            verbose=True)



# testing: helper function: plot predictions
def plot_predict(ts_actual, ts_test, ts_pred):
    
    # plot time series, limited to forecast horizon
    plt.figure(figsize=FIGSIZE)
    
    ts_actual.plot(label="actual price")                                       # plot actual
    # trade.plot(label="actual trade")
    ts_pred.plot(low_quantile=qL1, high_quantile=qU1, label=label_q1)    # plot U1 quantile band
    ts_pred.plot(low_quantile=qL2, high_quantile=qU2, label=label_q2)   # plot U2 quantile band
    ts_pred.plot(low_quantile=qL3, high_quantile=qU3, label=label_q3)    # plot U3 quantile band
    ts_pred.plot(central_quantile="mean", label="expected")              # plot "mean" or median=0.5

    plt.title("TFT: test set (MAPE: {:.2f}%)".format(mape(ts_test, ts_pred)))
    

    # fig, ax1 = plt.subplots()

    # ax2 = ax1.twinx()y
    # ax3 = ax1.twinx()

    # ax1.plot(ts_actual)
    # ts_pred = ts_pred.quantile_df()
    # ax2.plot(ts_pred, color = 'red')
    # ts_pred = TimeSeries.from_dataframe(ts_pred, fill_missing_dates=fill, freq=None)
    # ax3.plot(trade, color = 'green')
    # plt.title("TFT: test set (MAPE: {:.2f}%)".format(mape(ts_test, ts_pred)))

    # plt.legend()
    plt.show()


ts_pred = scalerP.inverse_transform(ts_tpred)
print(type(ts_pred))
plot_predict(ts, ts_test, ts_pred)
