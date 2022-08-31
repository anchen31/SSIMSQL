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
from darts.models import TransformerModel
from darts.metrics import mape, rmse
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression
from sklearn.linear_model import LinearRegression

import config

password = config.password

# 3.84 w standard scaler`

# min_max_scaler = preprocessing.MinMaxScaler()
# scaler = MinMaxScaler(feature_range = (0,1))
scaler = StandardScaler()

LOAD = False         # True = load previously saved model from disk?  False = (re)train the model
# SAVE = "/_TForm_model10e.pth.tar"   # file name to save the model under

SAVE = "/_distrib_nn_pct10e.pth.tar"   # file name to save the model under

EPOCHS = 10
INLEN = 10          # input size
FEAT = 16           # d_model = number of expected features in the inputs, up to 512    
HEADS = 8           # default 8
ENCODE = 4          # encoder layers
DECODE = 4          # decoder layers
DIM_FF = 256         # dimensions of the feedforward network, default 2048
BATCH = 4           # batch size
ACTF = "gelu"       # activation function, relu (default) or gelu
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

#8.44 mape for 50%

SPLIT = 0.8         # train/test %

FIGSIZE = (9, 6)


qL1, qL2, qL3 = 0.01, 0.05, 0.10        # percentiles of predictions: lower bounds
qU1, qU2, qU3 = 1-qL1, 1-qL2, 1-qL3     # upper bounds derived from lower bounds
label_q1 = f'{int(qU1 * 100)} / {int(qL1 * 100)} percentile band'
label_q2 = f'{int(qU2 * 100)} / {int(qL2 * 100)} percentile band'
label_q3 = f'{int(qU3 * 100)} / {int(qL3 * 100)} percentile band'

mpath = os.path.abspath(os.getcwd()) + SAVE     # path and file name to save the model


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

def get_data(start_date, end_date):
  name = 'SPY'
  ticker = yfinance.Ticker(name)
  data = ticker.history(interval="1d",start=start_date, end=end_date)
  data['Date'] = pd.to_datetime(data.index)
  data['Date'] = data['Date'].apply(mpl_dates.date2num)
  data = data.loc[:,['Date', 'Open', 'High', 'Low', 'Close']]
  data['RSI'] = ta.RSI(data['Close'], timeperiod=14)
  data['crossover'] = (data['Close'].ewm(12).mean() - data['Close'].ewm(26).mean()).ewm(span=9, adjust=False, min_periods=9).mean()
  data['1'] = data['Close'].pct_change(1).shift(-1)
  data['2'] = data['Close'].pct_change(2).shift(-2)
  data['3'] = data['Close'].pct_change(3).shift(-3)
  data['4'] = data['Close'].pct_change(4).shift(-4)
  data['5'] = data['Close'].pct_change(5).shift(-5)
  data['6'] = data['Close'].pct_change(6).shift(-6)
  data['7'] = data['Close'].pct_change(7).shift(-7)
  data['8'] = data['Close'].pct_change(8).shift(-8)
  data['9'] = data['Close'].pct_change(9).shift(-9)
  data['10'] = data['Close'].pct_change(10).shift(-10)
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
  data['count'] = range(len(data))

  # data = data.dropna()

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
  # data = data.set_index('Close')

  j = 0
  length = length - 1
  while j <= length:
    selected = data[(data['RSI'] <= rsi[j]+1) & (data['RSI'] >= rsi[j]-1)]
    # print(len(selected))
    selected = selected[(selected['STCH'] <= (Stoch[j]+1)) & (selected['STCH'] >= (Stoch[j]-1))]
    # print(len(selected))
    selected = selected[((selected['MACD'] > 0) & (MACD[j] > 0)) | ((selected['MACD'] < 0) & (MACD[j] < 0))]
    # print(len(selected))

    selected['count'] = selected['count']+j
    
    selected = selected.iloc[:,7:17-j]
    
    # print(selected.columns)
    total_data = total_data.append(selected)
    j+=1



  total_data = pd.DataFrame(justify(total_data.values, invalid_val=np.nan, axis=1, side='left'))

  return total_data

def graph(total_data):
  fig, ax = plt.subplots(2,5, figsize= (15,8))

  ax[0,0].hist(total_data[0], label='1', bins=30)
  ax[0,1].hist(total_data[1], label='2', bins=30)
  ax[0,2].hist(total_data[2], label='3', bins=30)
  ax[0,3].hist(total_data[3], label='4', bins=30)
  ax[0,4].hist(total_data[4], label='5', bins=30)
  ax[1,0].hist(total_data[5], label='6', bins=30)
  ax[1,1].hist(total_data[6], label='7', bins=30)
  ax[1,2].hist(total_data[7], label='8', bins=30)
  ax[1,3].hist(total_data[8], label='9', bins=30)
  ax[1,4].hist(total_data[9], label='10', bins=30)
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


def t_data():
  today = date.today()
  name = 'SPY'
  ticker = yfinance.Ticker(name)
  data = ticker.history(interval="1d",start="2010-1-15", end=today.strftime("%Y-%m-%d"))
  data['Date'] = pd.to_datetime(data.index)
  data['Date'] = data['Date'].apply(mpl_dates.date2num)
  data = data.loc[:,['Date', 'Open', 'High', 'Low', 'Close']]
  return data

tdata = t_data()


# gets the RSI, MACD, STOCH scores for data
# days is the days to look behind, d is the date selected
def get_values(days, d):
  total_data = pd.DataFrame()
  # today = date.today()
  # name = 'SPY'
  # ticker = yfinance.Ticker(name)
  # data = ticker.history(interval="1d",start="2010-1-15", end=today.strftime("%Y-%m-%d"))
  # data['Date'] = pd.to_datetime(data.index)
  # data['Date'] = data['Date'].apply(mpl_dates.date2num)
  # data = data.loc[:,['Date', 'Open', 'High', 'Low', 'Close']]
  data = tdata

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

  total_data = total_data[['RSI', 'STCH', 'MACD']]

  return total_data


def backtest(dates, k, data):
  df = pd.DataFrame()
  l = []

  for j in dates.index.values:
    d = get_values(10, j)
    rsi = d.RSI.values.tolist()
    Stoch = d.STCH.values.tolist()
    MACD = d.MACD.values.tolist()
    length = len(d)  

    total_data = manipulate_data(length, rsi, Stoch, MACD, data)
    # gets the first prediction
    total_data = total_data[k]

    pos_count = 0
    neg_count = 0
    pos_amt = 0
    neg_amt = 0

    for i in total_data:
      if i > 0:
        pos_count += i
        pos_amt +=1

      elif i < 0:
        neg_count += i
        neg_amt += 1
      else:
        pass

    try:
      l.append((((pos_count/pos_amt)*(pos_amt/(pos_amt + neg_amt))))+(((neg_count/neg_amt)*(neg_amt/(pos_amt + neg_amt)))))
    except:
      l.append(0)


  df.index = dates.index
  df['ratio'] = l
  # df['Close'] = dates.Close
  # df['1'] = dates['1']
  # df['ratio'] = l
  return df

def get_csv():
  data = get_data("2003-1-15", "2022-5-15")
  dates = get_data('2018-1-1', '2022-8-11')

  b1 = backtest(dates, 0, data)
  b2 = backtest(dates, 1, data)
  b3 = backtest(dates, 2, data)
  b4 = backtest(dates, 3, data)
  b5 = backtest(dates, 4, data)
  b6 = backtest(dates, 5, data)
  b7 = backtest(dates, 6, data)
  b8 = backtest(dates, 7, data)
  b9 = backtest(dates, 8, data)
  b10 = backtest(dates, 9, data)

  dates['ratio1'] = ta.RSI(b1.ratio, timeperiod=14)
  dates['ratio2'] = ta.RSI(b2.ratio, timeperiod=14)
  dates['ratio3'] = ta.RSI(b3.ratio, timeperiod=14)
  dates['ratio4'] = ta.RSI(b4.ratio, timeperiod=14)
  dates['ratio5'] = ta.RSI(b5.ratio, timeperiod=14)
  dates['ratio6'] = ta.RSI(b6.ratio, timeperiod=14)
  dates['ratio7'] = ta.RSI(b7.ratio, timeperiod=14)
  dates['ratio8'] = ta.RSI(b8.ratio, timeperiod=14)
  dates['ratio9'] = ta.RSI(b9.ratio, timeperiod=14)
  dates['ratio10'] = ta.RSI(b10.ratio, timeperiod=14)

  dates.to_csv('NNdistribuition_data.csv', index=True)
  print('Done')


def plot_predict(ts_actual, ts_test, ts_pred, trade):

  fig1, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

  ax1.plot(ts_actual, label="price", c='g')

  ax3.plot(trade, label="actual trade", c='b')

  lol = ts_pred.quantile_df(quantile=0.5)

  N = int(len(lol))

  lol = lol.set_index(ts_actual.index[-N:])

  ax2.plot(lol, label="prediction")

  # ts_pred.plot(low_quantile=qL1, high_quantile=qU1, label=label_q1)    # plot U1 quantile band
  # #ts_pred.plot(low_quantile=qL2, high_quantile=qU2, label=label_q2)   # plot U2 quantile band
  # ts_pred.plot(low_quantile=qL3, high_quantile=qU3, label=label_q3)    # plot U3 quantile band
  # ts_pred.plot(central_quantile="mean", label="expected")              # plot "mean" or median=0.5

  ax1.title.set_text("TFT: test set (MAPE: {:.2f}%)".format(mape(ts_test, ts_pred)))
  ax1.legend()
  ax2.legend()

  # plt.xticks(dates.index, dates.values)
  # plt.gcf().autofmt_xdate()

  plt.show()




def main():
  d = pd.read_csv('NNdistribuition_data.csv')
  # print(len(d))
  d = d.dropna()
  d = d.reset_index(drop=True)

  new_df = d[['Date', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']].copy()

  df = new_df.copy()
  df = df.drop(columns=['Date'])  

  # plt.figure(figsize = (15,15))
  # sns.set(font_scale=0.75)
  # ax = sns.heatmap(df.corr().round(3), 
  #             annot=True, 
  #             square=True, 
  #             linewidths=.75, cmap="coolwarm", 
  #             fmt = ".2f", 
  #             annot_kws = {"size": 11})
  # ax.xaxis.tick_bottom()
  # plt.title("correlation matrix")
  # plt.show()


  fill = False
  freq = None

  ts_P = TimeSeries.from_series(df["1"], fill_missing_dates=fill, freq=freq)
  ts_P = ts_P.pd_dataframe()
  ts_P_1 = ts_P.fillna(method='bfill')
  ts_P = TimeSeries.from_series(ts_P_1)

  # creates time series object covariate feature, This is multivariate
  df_covF = df.loc[:, df.columns != "1"]
  ts_covF = TimeSeries.from_dataframe(df_covF, fill_missing_dates=fill, freq=freq)
  ts_covF = ts_covF.pd_dataframe()
  ts_covF_1 = ts_covF.fillna(method='bfill')
  ts_covF = TimeSeries.from_series(ts_covF_1)



  ts_train, ts_test = ts_P.split_after(SPLIT)

  scalerP = Scaler(scaler)
  scalerP.fit_transform(ts_P)
  ts_ttrain = scalerP.transform(ts_train)
  ts_ttest = scalerP.transform(ts_test)    
  ts_t = scalerP.transform(ts_P)

  # train/test split and scaling of FEATURE covariates
  covF_train, covF_test = ts_covF.split_after(SPLIT)

  scalerF = Scaler(scaler)
  scalerF.fit_transform(ts_covF)
  covF_ttrain = scalerF.transform(covF_train) 
  covF_ttest = scalerF.transform(covF_test)   
  covF_t = scalerF.transform(ts_covF)

  model = TransformerModel(
                      input_chunk_length = INLEN,
                      output_chunk_length = N_FC,
                      batch_size = BATCH,
                      n_epochs = EPOCHS,
                      model_name = "Transformer_price",
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
                      add_encoders={"cyclic": {"future": ["dayofweek", "month"]}},
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


  ts_pred = scalerP.inverse_transform(ts_tpred)

  ts = d.Open
  pct = d['1']
  plot_predict(ts, ts_test, ts_pred, pct)


if __name__== '__main__':
    # put the loop logic here eto loop through everything accordingly
    main()