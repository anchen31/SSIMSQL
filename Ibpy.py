# This file proccess all of the data to be sent to model py in a df

from ib_insync import *
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import time
import math
import bisect
import pymysql
import config
pymysql.install_as_MySQLdb()
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from ta.utils import dropna
from ta.volatility import BollingerBands
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt

barSze = '1 day'
durStrng = '4 Y'
#0.6 ms difference but can gather more data for the openning bars
RTH = True
list1 = ['GLD', 'UVXY', 'SQQQ', 'CVX', 'RIO', 'NUE', 'LWAY', 'TSN', 'NTR', 'ADM', 'HYG', 'SRLN', 'JNK', 'EWH', 'GBTC', 'USO', 'DIA', 'QQQ', 'IWM', 'IEF', 'SIVR', 'FXB', 'FXE']

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=2)


def datafrm():
    contract1 = Stock('SPY', 'SMART', 'USD')
    barsList = []

    bars = ib.reqHistoricalData(
        contract1, 
        endDateTime='',
        durationStr=durStrng,
        barSizeSetting=barSze,
        whatToShow='TRADES',
        useRTH=RTH,
        formatDate=1,
        keepUpToDate=False)

    barsList.append(bars)

    allBars = [b for bars in reversed(barsList) for b in bars]
    df = util.df(allBars)
    return df


# create giant df here to add

# function that takes in a bunch of list of strings of tickers and 
def getData(list):
    data1 = datafrm()
    for i in list:
        contract = Stock(i, 'SMART', 'USD')

        barsList = []

        bars = ib.reqHistoricalData(
            contract, 
            endDateTime='',
            durationStr=durStrng,
            barSizeSetting=barSze,
            whatToShow='TRADES',
            useRTH=RTH,
            formatDate=1,
            keepUpToDate=False)

        barsList.append(bars)

        allBars = [b for bars in reversed(barsList) for b in bars]
        df = util.df(allBars)
        df[i] = df['close']
        df = df[['date', i]]

        data = pd.merge(data1, df, on=['date'])

        data1 = data

    return data

levels = []

def RoudUp(x, base=5):
    return base * math.ceil(x/base)

def RoudDown(x, base=5):
    return base * math.floor(x/base)

def isFarFromLevel(l):
    return np.sum([abs(l-x) < s  for x in levels]) == 0

def isSupport(df,i):
    support = df['low'][i] < df['low'][i-1]  and df['low'][i] < df['low'][i+1] and df['low'][i+1] < df['low'][i+2] and df['low'][i-1] < df['low'][i-2]
    return support


def isResistance(df,i):
    resistance = df['high'][i] > df['high'][i-1]  and df['high'][i] > df['high'][i+1] and df['high'][i+1] > df['high'][i+2] and df['high'][i-1] > df['high'][i-2]
    return resistance 

# takes in a List and finds the two values next to b, one value is higher if it exist and one value is lower
def closest(lst, b):
    lst = list(lst)
    lst.sort()
    n, j = len(lst), bisect.bisect_left(lst, b)
    if b < lst[-1]:
        if lst[j] > b:
           return (None if j == 0 else lst[j-1]), lst[j] 
        else:
           return lst[j], (None if j >= n - 1 else lst[j + 1])
    else:
        return lst[len(lst)-1], None



    n, j = len(lst), bisect.bisect_left(lst, b)
    return ((None if j == 0 else lst[j-1]), lst[j]) if lst[j] > b else (lst[j], (None if j >= n - 1 else lst[j + 1]))
 
# Calculates the long term support/resistance and puts it into a list
def ltSR():
    contract1 = Stock('SPY', 'SMART', 'USD')
    barsList = []

    bars = ib.reqHistoricalData(
        contract1, 
        endDateTime='',
        durationStr='8 Y',
        barSizeSetting='1 day',
        whatToShow='TRADES',
        useRTH=RTH,
        formatDate=1,
        keepUpToDate=False)

    barsList.append(bars)

    allBars = [b for bars in reversed(barsList) for b in bars]
    df = util.df(allBars)

    s =  np.mean(df['high'] - df['low'])

    levels = []
    for i in range(2,df.shape[0]-2):
      if isSupport(df,i):
        l = df['low'][i]

        if isFarFromLevel(l):
          # levels.append((i,l))
          levels.append(l)

      elif isResistance(df,i):
        l = df['high'][i]

        if isFarFromLevel(l):
          # levels.append((i,l))
          levels.append(l)

    return levels


# does all of the ta stuff and puts it into a mysql db
def main():

    counter = 0

    while True:
        now = datetime.now()

        #pulls data after each minute and
        while now.second != 1:
            time.sleep(1)
            print(now.second)
            now = datetime.now()

        df = getData(list1)

        indicator_bb = BollingerBands(close=df["close"], window=20, window_dev=2)

        df['bb_bbm'] = round(indicator_bb.bollinger_mavg(), 4)
        df['bb_bbh'] = round(indicator_bb.bollinger_hband(), 4)
        df['bb_bbl'] = round(indicator_bb.bollinger_lband(), 4)

        v = df['volume']
        p = df['close']

        df['VWAP'] = round(((v * p).cumsum() / v.cumsum()), 4)


        delta = df['close'].diff()
        up = delta.clip(lower=0)
        down = -1*delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rs = ema_up/ema_down

        df['RSI'] = round(100-(100/(1 + rs)), 4)

        # LT AND ST S/R ##########################################################################################################################################
        s =  np.mean(df['high'] - df['low'])

        levels = []
        for i in range(2,df.shape[0]-2):
          if isSupport(df,i):
            l = df['low'][i]

            if isFarFromLevel(l):
              # levels.append((i,l))
              levels.append(l)

          elif isResistance(df,i):
            l = df['high'][i]

            if isFarFromLevel(l):
              # levels.append((i,l))
              levels.append(l)

        # Stores it into dataframe
        LTe = ltSR()
        df['STsupp'] = 0
        df['STres'] = 0
        df['LTsupp'] = 0
        df['LTres'] = 0

        for ind in df.index:
            price = df['close'][ind]
            ST = closest(levels, price)
            LT = closest(LTe, price)

            #This will take care of Nan Values on the S/R
            if(ST[0] == None):
                df.loc[ind, ['STsupp']] = RoudDown(price)
            else:
                df.loc[ind, ['STsupp']] = ST[0]

            if(ST[1] == None):
                df.loc[ind, ['STres']] = RoudUp(price)
            else:
                df.loc[ind, ['STres']] = ST[1]

            if(LT[0] == None):
                df.loc[ind, ['LTsupp']] = RoudDown(price)
            else:
                df.loc[ind, ['LTsupp']] = LT[0]

            if(LT[1] == None):
                df.loc[ind, ['LTres']] = RoudUp(price)
            else:
                df.loc[ind, ['LTres']] = LT[1]

        #print(df.columns)

        print(df.columns)

        df.to_csv('four_year_date.csv', index=False)

        #################################################Create a new db for this data, this will be the main db that will have everything else join it###
        # engine = create_engine(config.engine)
        # ############################## Create config.engine1 that has a different db loaction #######################
        # with engine.begin() as connection:
        #     # maybe "replace" fixes the missing values problem?
        #     df.to_sql(name='ibpy', con=connection, if_exists='replace', index=False)

    ib.disconnect()

#df = dropna(df)


# i can just do the df into a sql thingy again lol



###########################################################################################################################

if __name__== '__main__':
    # put the loop logic here eto loop through everything accordingly
    main()


#################################################################### WHY IS BENZINGA DATA 250$ LOL #####################################################################

