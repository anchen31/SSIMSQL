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
from datetime import date
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

def get_macd(price, slow, fast, smooth):
    exp1 = price.ewm(span = fast, adjust = False).mean()
    exp2 = price.ewm(span = slow, adjust = False).mean()
    macd = pd.DataFrame(exp1 - exp2)
    signal = pd.DataFrame(macd.ewm(span = smooth, adjust = False).mean())
    hist = macd - signal
    return hist

def get_data():
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
  data['MACD'] = get_macd(data['Close'], 26, 12, 9)

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

  return data


# calculate the expected value of gaining or losing
def show_results(total_data):
  k = 0
  while k < 10:
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

# combines previous probabilities with future probabilities
def manipulate_data(length, rsi, Stoch, MACD, data):
  total_data = pd.DataFrame()

  j = 0
  while j <= length:
    selected = data[(data['RSI'] <= rsi[j]+1) & (data['RSI'] >= rsi[j]-1)]
    # print(len(selected))
    selected = selected[(selected['STCH'] <= (Stoch[j]+1)) & (selected['STCH'] >= (Stoch[j]-1))]
    # print(len(selected))
    selected = selected[((selected['MACD'] > 0) & (selected['MACD'] > MACD[j])) | ((selected['MACD'] < 0) & (selected['MACD'] < MACD[j]))]
    # print(len(selected))

    selected = selected.iloc[:,7+j:17]
    total_data = total_data.append(selected)
    j+=1

  # print(total_data.head(60))

  total_data = pd.DataFrame(justify(total_data.values, invalid_val=np.nan, axis=1, side='left'))

  return total_data

def graph(total_data):
  fig, ax = plt.subplots(2,5, figsize= (18,10))

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
  plt.show()

# gets the RSI, MACD, STOCH scores for data
def get_dates(days, d):
  total_data = pd.DataFrame()
  today = date.today()
  name = 'SPY'
  ticker = yfinance.Ticker(name)
  data = ticker.history(interval="1d",start="2010-1-15", end=today.strftime("%Y-%m-%d"))
  data['Date'] = pd.to_datetime(data.index)
  data['Date'] = data['Date'].apply(mpl_dates.date2num)
  data = data.loc[:,['Date', 'Open', 'High', 'Low', 'Close']]

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

  data['RSI'] = ta.RSI(data['Close'], timeperiod=14)
  data['MACD'] = get_macd(data['Close'], 26, 12, 9)
  data['Date'] = range(len(data))

  d1 = data.loc[data.index == d]
  d1 = d1['Date'].values[0]

  i = d1

  while i > (d1-days):
    row = data.loc[data['Date'] == i]
    total_data = total_data.append(row)
    i-=1

  

  print(total_data)

# data = pd.read_csv('disData.csv')
# data.to_csv('disData.csv', index=False)
# print(pd.read_csv('disData.csv'))
# print(data)

################################################################################################# getting data stuff



def main():
  # 5/13/22 latest
  data = get_data()

  # date 5/24
  # rsi = [39.74, 41.25, 35.82, 35.69, 36.65, 44.26]
  # Stoch = [3.74, 12.84, 4.09, -15.92, -18.49, 11.11]
  # MACD = [1, -1, -1, -1, -1, -1]
  # date 77/29
  # rsi = [65.91, 62.55, 59.39, 51.77, 56.29, 55.92]
  # Stoch = [.95, 11.19, 12.05, -10.16, -4.66, -9.62]
  # date 8/1
  # rsi = [64.68, 65.91, 62.55, 59.39, 51.77, 56.29]
  # Stoch = [-3.51, .95, 11.19, 12.05, -10.16, -4.66]
  # 8/2
  # rsi = [62.25, 64.68, 65.91, 62.55, 59.39, 51.77, 56.29, 55.92]
  # Stoch = [-4.77, -3.51, .95, 11.19, 12.05, -10.16, -4.66, -9.62]
  # MACD = [1, 1, 1, 1, 1, 1, 1, 1]

  rsi = [62.25, 64.68, 65.91, 62.55, 59.39]
  Stoch = [-4.77, -3.51, .95, 11.19, 12.05]
  MACD = [1, 1, 1, 1, 1]
  length = 4


  get_dates(4, '2016-1-15')

  # total_data = manipulate_data(length, rsi, Stoch, MACD, data)
  # show_results(total_data)
  # graph(total_data)


if __name__== '__main__':
    # put the loop logic here eto loop through everything accordingly
    main()

# print(total_data)




