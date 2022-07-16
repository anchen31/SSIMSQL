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

spy = pd.DataFrame()

spy['open'] = df['open']

spy['SMA10'] = df.open.rolling(10).mean()/df.open
spy['SMA20'] = df.open.rolling(20).mean()/df.open
spy['EMA10'] = df.open.ewm(span=10, adjust=False).mean()/df.open
spy['EMA20'] = df.open.ewm(span=20, adjust=False).mean()/df.open
spy['SMADiff'] = spy.SMA20-spy.SMA10
spy['EMADiff'] = spy.EMA20-spy.EMA10


spy = spy.dropna(axis=0)


sma10 = spy['SMA10'].describe()
sma20 = spy['SMA20'].describe()
ema10 = spy['EMA10'].describe()
ema20 = spy['EMA20'].describe()
smadiff = spy['SMADiff'].describe()
emadiff = spy['EMADiff'].describe()



# calculate the pct diff or the point diff?

fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(16, 10))





ax[0,0].hist(spy['SMA10'], bins=100)
ax[0,0].axvline(sma10[1], color='r', linestyle='solid', linewidth=1, label="mean")
ax[0,0].axvline(sma10[4], color='g', linestyle='solid', linewidth=1, label='25%')
ax[0,0].axvline(sma10[5], color='b', linestyle='solid', linewidth=1, label='50%')
ax[0,0].axvline(sma10[6], color='g', linestyle='solid', linewidth=1, label='75%')
ax[0,0].axvline(spy['SMA10'].quantile(0.95), color='y', linestyle='solid', linewidth=1, label='95%')
ax[0,0].axvline(spy['SMA10'].quantile(0.05), color='y', linestyle='solid', linewidth=1, label='5%')
ax[0,0].set_title('SMA10')
ax[0,0].legend(loc='upper right')

ax[1,0].hist(spy['SMA20'], bins=100)
ax[1,0].axvline(sma20[1], color='r', linestyle='solid', linewidth=1, label="mean")
ax[1,0].axvline(sma20[4], color='g', linestyle='solid', linewidth=1, label='25%')
ax[1,0].axvline(sma20[5], color='b', linestyle='solid', linewidth=1, label='50%')
ax[1,0].axvline(sma20[6], color='g', linestyle='solid', linewidth=1, label='75%')
ax[1,0].axvline(spy['SMA20'].quantile(0.95), color='y', linestyle='solid', linewidth=1, label='95%')
ax[1,0].axvline(spy['SMA20'].quantile(0.05), color='y', linestyle='solid', linewidth=1, label='5%')
ax[1,0].set_title('SMA20')

ax[0,1].hist(spy['EMA10'], bins=100)
ax[0,1].axvline(ema10[1], color='r', linestyle='solid', linewidth=1, label="mean")
ax[0,1].axvline(ema10[4], color='g', linestyle='solid', linewidth=1, label='25%')
ax[0,1].axvline(ema10[5], color='b', linestyle='solid', linewidth=1, label='50%')
ax[0,1].axvline(ema10[6], color='g', linestyle='solid', linewidth=1, label='75%')
ax[0,1].axvline(spy['EMA10'].quantile(0.95), color='y', linestyle='solid', linewidth=1, label='95%')
ax[0,1].axvline(spy['EMA10'].quantile(0.05), color='y', linestyle='solid', linewidth=1, label='5%')
ax[0,1].set_title('EMA10')

ax[1,1].hist(spy['EMA20'], bins=100)
ax[1,1].axvline(ema20[1], color='r', linestyle='solid', linewidth=1, label="mean")
ax[1,1].axvline(ema20[4], color='g', linestyle='solid', linewidth=1, label='25%')
ax[1,1].axvline(ema20[5], color='b', linestyle='solid', linewidth=1, label='50%')
ax[1,1].axvline(ema20[6], color='g', linestyle='solid', linewidth=1, label='75%')
ax[1,1].axvline(spy['EMA20'].quantile(0.95), color='y', linestyle='solid', linewidth=1, label='95%')
ax[1,1].axvline(spy['EMA20'].quantile(0.05), color='y', linestyle='solid', linewidth=1, label='5%')
ax[1,1].set_title('EMA20')

ax[2,0].hist(spy['SMADiff'], bins=100)
ax[2,0].axvline(smadiff[1], color='r', linestyle='solid', linewidth=1, label="mean")
ax[2,0].axvline(smadiff[4], color='g', linestyle='solid', linewidth=1, label='25%')
ax[2,0].axvline(smadiff[5], color='b', linestyle='solid', linewidth=1, label='50%')
ax[2,0].axvline(smadiff[6], color='g', linestyle='solid', linewidth=1, label='75%')
ax[2,0].axvline(spy['SMADiff'].quantile(0.95), color='y', linestyle='solid', linewidth=1, label='95%')
ax[2,0].axvline(spy['SMADiff'].quantile(0.05), color='y', linestyle='solid', linewidth=1, label='5%')
ax[2,0].set_title('SMADiff')

ax[2,1].hist(spy['EMADiff'], bins=100)
ax[2,1].axvline(emadiff[1], color='r', linestyle='solid', linewidth=1, label="mean")
ax[2,1].axvline(emadiff[4], color='g', linestyle='solid', linewidth=1, label='25%')
ax[2,1].axvline(emadiff[5], color='b', linestyle='solid', linewidth=1, label='50%')
ax[2,1].axvline(emadiff[6], color='g', linestyle='solid', linewidth=1, label='75%')
ax[2,1].axvline(spy['EMADiff'].quantile(0.95), color='y', linestyle='solid', linewidth=1, label='95%')
ax[2,1].axvline(spy['EMADiff'].quantile(0.05), color='y', linestyle='solid', linewidth=1, label='5%')
ax[2,1].set_title('EMADiff')


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


