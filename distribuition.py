import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
import math
import random
import time
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import norm
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
from sklearn.linear_model import LinearRegression


df = pd.read_csv('four_year_date.csv')
df = df.dropna(axis=0)

print(df.columns)

df1 = df.copy()
df1 = df1.drop(columns=['date'])

for i in df1.columns:
  # does the linear regression on the columns
  X = df1.index.values
  y = df1[[i]].values

  length = len(X)

  X = X.reshape(length, 1)
  y = y.reshape(length, 1)

  regressor = LinearRegression()
  regressor.fit(X, y)

  y_pred1 = regressor.predict(X)
  y_pred = pd.DataFrame(y_pred1)

  # create a new value based off of 
  df1[i] = df1[i] - y_pred[0]

df = df.set_index('date')

df1.index = df.index.astype('datetime64[ns]')

spy = pd.DataFrame()
# spy.index = df['date']

spy['open'] = df['open']

spy['SMA10'] = df.open.rolling(3).mean()/df.open
spy['SMA20'] = df.open.rolling(60).mean()/df.open
spy['EMA10'] = df.open.ewm(span=10, adjust=False).mean()/df.open
spy['EMA20'] = df.open.ewm(span=10, adjust=False).mean()/df.open
spy['SMADiff'] = spy.SMA20-spy.SMA10
spy['EMADiff'] = spy.EMA20-spy.EMA10








# spy['totalDiff'] = spy['EMADiff']-spy['SMADiff']

spy1 = spy.copy()
spy1.index = df.index.astype('datetime64[ns]')
spy1 = spy1.dropna(axis=0)


spy = spy.dropna(axis=0)


sma10 = spy['SMA10'].describe()
sma20 = spy['SMA20'].describe()
ema10 = spy['EMA10'].describe()
ema20 = spy['EMA20'].describe()
smadiff = spy['SMADiff'].describe()
emadiff = spy['EMADiff'].describe()




fig, (ax, ax2) = plt.subplots(2, sharex=True)

upper = 0.9
lower = 0.1

spycp = spy1[spy1['open'] > spy1['open'].quantile(upper)]
spycn = spy1[spy1['open'] < spy1['open'].quantile(lower)]

count_pos = spy1[spy1['SMA10'] > spy1['SMA10'].quantile(upper)]
count_neg = spy1[spy1['SMA10'] < spy1['SMA10'].quantile(lower)]

count_pos0 = spy1[spy1['EMA20'] > spy1['EMA20'].quantile(upper)]
count_neg0 = spy1[spy1['EMA20'] < spy1['EMA20'].quantile(lower)]

count_pos1 = spy1[spy1['SMADiff'] > spy1['SMADiff'].quantile(upper)]
count_neg1 = spy1[spy1['SMADiff'] < spy1['SMADiff'].quantile(lower)]

count_pos2 = spy1[spy1['EMADiff'] > spy1['EMADiff'].quantile(upper)]
count_neg2 = spy1[spy1['EMADiff'] < spy1['EMADiff'].quantile(lower)]


ax.scatter(x=spycp.index, y=spycp['open'], c='g', label='sma10pos')
ax.scatter(x=spycn.index, y=spycn['open'], c='r', label='sma10pos')
# ax.scatter(x=count_pos.index, y=count_pos['open'], c='r', label='sma10pos')
# ax.scatter(x=count_pos1.index, y=count_pos1['open'], c='rosybrown', label='smadiffpos')
# ax.scatter(x=count_pos2.index, y=count_pos2['open'], c='maroon', label='emadiffpos')
# ax.scatter(x=count_pos0.index, y=count_pos0['open'], c='mistyrose', label='ema20pos')
# ax.scatter(x=count_neg.index, y=count_neg['open'], c='mediumspringgreen', label='sma10neg')
# ax.scatter(x=count_neg1.index, y=count_neg1['open'], c='darkgreen', label='smadiffneg')
# ax.scatter(x=count_neg2.index, y=count_neg2['open'], c='limegreen', label='emadiffneg')
# ax.scatter(x=count_neg0.index, y=count_neg0['open'], c='lightseagreen', label='ema20neg')
ax.plot(spy1['open'])
ax2.plot(spy1['EMA10'], c='r', label='sma10')
ax2.plot(spy1['EMA20'], c='g', label='sma20')
ax2.legend()
ax.legend()






# plt.plot(spy1['open'])
# plt.plot(count_neg)
# plt.plot(count_pos)



# spy.hist('SMA10', ax=axes[0], bins=100)
# spy.hist('SMA20', ax=axes[1], bins=100)
# spy.hist('EMA20', ax=axes[2], bins=100)
# spy.hist('EMA20', ax=axes[3], bins=100)
# spy.hist('SMADiff', ax=axes[4], bins=100)
# spy.hist('EMADiff', ax=axes[5], bins=100)
# plt.plot(spy['SMA10'].mean(), color='k', linestyle='dashed', linewidth=1)


# spy.hist(['SMA10', 'SMA20', 'EMA10', 'EMA20', 'SMADiff', 'EMADiff'], bins=150)
# plt.axvline(spy['SMA10'].mean(), color='k', linestyle='dashed', linewidth=1)
# plt.hist(spy[['SMA10', 'SMA20', 'EMA10', 'EMA20']], bins=100, label=['SMA10', 'SMA20', 'EMA10', 'EMA20'])
# plt.legend(loc='upper right')

plt.show()


# print(spy)


