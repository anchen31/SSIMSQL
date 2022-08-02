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
import talib as ta
import yfinance
import matplotlib.dates as mpl_dates
# from sqlalchemy import create_engine
# import pymysql
# pymysql.install_as_MySQLdb()
# import mysql.connector
# from mysql.connector import Error

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
# from darts.models import TransformerModel
from darts.metrics import mape, rmse
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression
from sklearn.linear_model import LinearRegression

def justify(a, invalid_val=0, axis=1, side='left'):    
    """
    Justifies a 2D array

    Parameters
    ----------
    A : ndarray
        Input array to be justified
    axis : int
        Axis along which justification is to be made
    side : str
        Direction of justification. It could be 'left', 'right', 'up', 'down'
        It should be 'left' or 'right' for axis=1 and 'up' or 'down' for axis=0.

    """

    if invalid_val is np.nan:
        mask = ~np.isnan(a)
    else:
        mask = a!=invalid_val
    justified_mask = np.sort(mask,axis=axis)
    if (side=='up') | (side=='left'):
        justified_mask = np.flip(justified_mask,axis=axis)
    out = np.full(a.shape, invalid_val) 
    if axis==1:
        out[justified_mask] = a[mask]
    else:
        out.T[justified_mask.T] = a.T[mask.T]
    return out

################################################################################################# getting data stuff
name = 'SPY'
ticker = yfinance.Ticker(name)
data = ticker.history(interval="1d",start="2004-1-15", end="2022-05-15")
data['Date'] = pd.to_datetime(data.index)
data['Date'] = data['Date'].apply(mpl_dates.date2num)
data = data.loc[:,['Date', 'Open', 'High', 'Low', 'Close']]
data['RSI'] = ta.RSI(data['Close'], timeperiod=14)
data['crossover'] = (data['Close'].ewm(12).mean() - data['Close'].ewm(26).mean()).ewm(span=9, adjust=False, min_periods=9).mean()
data['pct_change_closing1'] = data['Close'].pct_change().shift(-1)
data['pct_change_closing2'] = data['Close'].pct_change(2).shift(-2)
data['pct_change_closing3'] = data['Close'].pct_change(3).shift(-3)
data['pct_change_closing4'] = data['Close'].pct_change(4).shift(-4)
data['pct_change_closing5'] = data['Close'].pct_change(5).shift(-5)
data['pct_change_closing6'] = data['Close'].pct_change(6).shift(-6)
data['pct_change_closing7'] = data['Close'].pct_change(7).shift(-7)
data['pct_change_closing8'] = data['Close'].pct_change(8).shift(-8)
data['pct_change_closing9'] = data['Close'].pct_change(9).shift(-9)
data['pct_change_closing10'] = data['Close'].pct_change(10).shift(-10)

k_period = 14
d_period = 3

# Adds a "n_high" column with max value of previous 14 periods
data['n_high'] = data['High'].rolling(k_period).max()
# Adds an "n_low" column with min value of previous 14 periods
data['n_low'] = data['Low'].rolling(k_period).min()
# Uses the min/max values to calculate the %k (as a percentage) (Fast)
data['%K'] = (data['Close'] - data['n_low']) * 100 / (data['n_high'] - data['n_low'])
# Uses the %k to calculates a SMA over the past 3 values of %k (Slow)
data['%D'] = data['%K'].rolling(d_period).mean()
# Create new column for difference
data['STCH'] = data['%K'] - data['%D']

data = data.dropna()

# data = pd.read_csv('disData.csv')
# data.to_csv('disData.csv', index=False)
# print(pd.read_csv('disData.csv'))
# print(data)


total_data = pd.DataFrame()

################################################################################################# getting data stuff


# date 5/24
# rsi = [39.74, 41.25, 35.82, 35.69, 36.65, 44.26]
# Stoch = [3.74, 12.84, 4.09, -15.92, -18.49, 11.11]
# date 77/29
# rsi = [65.91, 62.55, 59.39, 51.77, 56.29, 55.92]
# Stoch = [.95, 11.19, 12.05, -10.16, -4.66, -9.62]
# date 8/1
rsi = [64.68, 65.91, 62.55, 59.39, 51.77, 56.29]
Stoch = [-3.51, .95, 11.19, 12.05, -10.16, -4.66]

#length of RSI and STOCH
length = 5


j = 0
while j <= length:
  selected = data[(data['RSI'] <= rsi[j]+1) & (data['RSI'] >= rsi[j]-1)]
  selected = selected[(selected['STCH'] <= (Stoch[j]+1)) & (selected['STCH'] >= (Stoch[j]-1))]

  selected = selected.iloc[:,7+j:17]
  total_data = total_data.append(selected)
  j+=1

# print(total_data.head(60))

total_data = pd.DataFrame(justify(total_data.values, invalid_val=np.nan, axis=1, side='left'))

# print(total_data)

# 5/13/22 latest

fig, ax = plt.subplots(2,5, figsize= (10,6))

ax[0,0].hist(total_data[0], label='pct_change_closing1')
ax[0,1].hist(total_data[1], label='pct_change_closing2')
ax[0,2].hist(total_data[2], label='pct_change_closing3')
ax[0,3].hist(total_data[3], label='pct_change_closing4')
ax[0,4].hist(total_data[4], label='pct_change_closing5')
ax[1,0].hist(total_data[5], label='pct_change_closing6')
ax[1,1].hist(total_data[6], label='pct_change_closing7')
ax[1,2].hist(total_data[7], label='pct_change_closing8')
ax[1,3].hist(total_data[8], label='pct_change_closing9')
ax[1,4].hist(total_data[9], label='pct_change_closing10')
ax[0,0].axvline(total_data[0].mean(), color='r', linestyle='dashed', linewidth=1)
ax[0,1].axvline(total_data[1].mean(), color='r', linestyle='dashed', linewidth=1)
ax[0,2].axvline(total_data[2].mean(), color='r', linestyle='dashed', linewidth=1)
ax[0,3].axvline(total_data[3].mean(), color='r', linestyle='dashed', linewidth=1)
ax[0,4].axvline(total_data[4].mean(), color='r', linestyle='dashed', linewidth=1)
ax[1,0].axvline(total_data[5].mean(), color='r', linestyle='dashed', linewidth=1)
ax[1,1].axvline(total_data[6].mean(), color='r', linestyle='dashed', linewidth=1)
ax[1,2].axvline(total_data[7].mean(), color='r', linestyle='dashed', linewidth=1)
ax[1,3].axvline(total_data[8].mean(), color='r', linestyle='dashed', linewidth=1)
ax[1,4].axvline(total_data[9].mean(), color='r', linestyle='dashed', linewidth=1)


# calculate the expected value of gaining or losing

k = 0
while k < 9:
  pos_count = 0
  neg_count = 0
  pos_amt = 0
  neg_amt = 0

  for i in total_data[k]:
    if i > 0:
      pos_count += i
      pos_amt +=1

    elif i < 0:
      neg_count += i
      neg_amt += 1
    else:
      pass


  print(k+1, " Days ahead")
  print('expected positive is: ', pos_count, ' Expected negative is: ', neg_count)
  print('amount positive is: ', pos_amt, ' amount negative is: ', neg_amt)
  print('Mean is: ', total_data[k].mean())
  try:
    print('expected pos value: ', (pos_count/pos_amt), 'expected neg Value: ', (neg_count/neg_amt))
    print('expected Value is: ', (((pos_count/pos_amt)*(pos_amt/(pos_amt + neg_amt))))+(((neg_count/neg_amt)*(neg_amt/(pos_amt + neg_amt)))))
  except:
    print("cannot divide by zero")

  print()
  k += 1



plt.show()



































df = pd.read_csv('four_year_date.csv')
df = df.dropna(axis=0)
df = df.set_index('date')
df.index = df.index.astype('datetime64[ns]')

# print(df.columns)

df1 = df.copy()

# for i in df1.columns:
#   # does the linear regression on the columns
#   X = df1.index.values
#   y = df1[[i]].values

#   length = len(X)

#   X = X.reshape(length, 1)
#   y = y.reshape(length, 1)

#   regressor = LinearRegression()
#   regressor.fit(X, y)

#   y_pred1 = regressor.predict(X)
#   y_pred = pd.DataFrame(y_pred1)

#   # create a new value based off of 
#   df1[i] = df1[i] - y_pred[0]


for i in df1.columns:
  df1[i] = ta.RSI(df1[i], timeperiod=14)

df1['RSI'] = df.RSI
df1['crsi'] = ta.RSI(df1['close'], timeperiod=13)

df1['rsiDiff'] = df['RSI'] - df1['crsi']
df1['price'] = df['open']




# spy['totalDiff'] = spy['EMADiff']-spy['SMADiff']
df1 = df1.dropna(axis=0)





fig, (ax, ax2) = plt.subplots(2, sharex=True)

upper = 0.95
lower = 0.05
ticker = 'GLD'



# spycp = spy1[spy1['open'] > spy1['open'].quantile(upper)]
# spycn = spy1[spy1['open'] < spy1['open'].quantile(lower)]

pos = df1[df1[ticker] > df1[ticker].quantile(upper)]
neg = df1[df1[ticker] < df1[ticker].quantile(lower)]




ax.scatter(x=pos.index, y=pos['price'], c='g', label='sma10pos')
ax.scatter(x=neg.index, y=neg['price'], c='r', label='sma10pos')


# ax.scatter(x=spyrsip.index, y=spyrsip['open'], c='g', label='sma10pos')
# ax.scatter(x=spyrsin.index, y=spyrsin['open'], c='r', label='sma10pos')
# ax.scatter(x=count_pos.index, y=count_pos['open'], c='r', label='sma10pos')
# ax.scatter(x=count_pos1.index, y=count_pos1['open'], c='rosybrown', label='smadiffpos')
# ax.scatter(x=count_pos2.index, y=count_pos2['open'], c='maroon', label='emadiffpos')
# ax.scatter(x=count_pos0.index, y=count_pos0['open'], c='mistyrose', label='ema20pos')
# ax.scatter(x=count_neg.index, y=count_neg['open'], c='mediumspringgreen', label='sma10neg')
# ax.scatter(x=count_neg1.index, y=count_neg1['open'], c='darkgreen', label='smadiffneg')
# ax.scatter(x=count_neg2.index, y=count_neg2['open'], c='limegreen', label='emadiffneg')
# ax.scatter(x=count_neg0.index, y=count_neg0['open'], c='lightseagreen', label='ema20neg')
ax.plot(df['open'])
ax2.plot(df['RSI'], c='orange', label='RSI')
ax2.plot(df1['crsi'], c='blue', label='crsi')
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

# plt.show()


# print(spy)


