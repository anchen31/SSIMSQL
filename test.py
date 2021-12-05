# import mysql.connector

# mydb = mysql.connector.connect(
#     host = 'localhost',
#     database='twitterdb', 
#     user='root', 
#     password = '@ndych3n1454L46i5Z9')

# mycursor = mydb.cursor(buffered=True)

# #mycursor.execute("SELECT * FROM TwitterSent")
# #mycursor.execute("SELECT * FROM tweetdb")
# #mycursor.execute("TRUNCATE TABLE TwitterSent")
# mycursor.execute("TRUNCATE TABLE tweetdb") 
# #// deletes all the data in table
# #mycursor.execute("DROP TABLE TwitterSent")
# #mycursor.execute("ALTER TABLE TwitterSent ADD sentiment VARCHAR(50)")
# #mycursor.execute("CREATE TABLE TwitterSent (timestamp_ms VARCHAR(20), sentiment VARCHAR(10))")

# #mycursor.execute("CREATE TABLE rednewsDB (timestamp_ms VARCHAR(20), reddit_sentiment VARCHAR(10), reddit_comm_sentiment VARCHAR(10), news_sentiment VARCHAR(10))")
# #mycursor.execute("DROP TABLE rednewsDB")
# #mycursor.execute("TRUNCATE TABLE rednewsDB")
# #mycursor.execute("SELECT * FROM rednewsDB")

# # mycursor.execute("CREATE TABLE IBPY (date VARCHAR(20), open VARCHAR(10), high VARCHAR(10), low VARCHAR(10), \
# #   close VARCHAR(10), volume VARCHAR(10), average VARCHAR(10), barCount VARCHAR(10), bb_bbm VARCHAR(10), bb_bbh VARCHAR(10), \
# #   bb_bbl VARCHAR(10), VWAP VARCHAR(10), RSI VARCHAR(10), STsupp VARCHAR(10), STres VARCHAR(10), LTsupp VARCHAR(10), LTres VARCHAR(10), \
# #   GLD VARCHAR(10), UVXY VARCHAR(10), SQQQ VARCHAR(10))")
# #mycursor.execute("DROP TABLE IBPY")
# #mycursor.execute("TRUNCATE TABLE IBPY")
# #mycursor.execute("SELECT * FROM IBPY")


# #myresult = mycursor.fetchall()

# #for x in myresult:
# #  print(x)

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


barSze = '10 mins'
durStrng = '1 Y'



ib = IB()
ib.connect('127.0.0.1', 7497, clientId=2)
#0.6 ms difference but can gather more data for the openning bars
RTH = True

sia = SentimentIntensityAnalyzer()

contract1 = Stock('SPY', 'SMART', 'USD')
GLD1 = Stock('GLD', 'SMART', 'USD')
UVXY1 = Stock('UVXY', 'SMART', 'USD')
SQQQ1 = Stock('SQQQ', 'SMART', 'USD')

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
    barsList = []

    bars = ib.reqHistoricalData(
        contract1, 
        endDateTime='',
        durationStr='1 Y',
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

# returns a df from ibkr highlighting it's price action
def datafrm():
    barsList = []

    ib.reqMktData(contract1, '', False, False)
    ticker = ib.ticker(contract1)
    ib.sleep(0.1)

    sPrice = ticker.marketPrice()

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

def GLD():
    barsList = []

    ib.reqMktData(GLD1, '', False, False)
    ticker = ib.ticker(GLD1)
    ib.sleep(0.1)

    sPrice = ticker.marketPrice()

    bars = ib.reqHistoricalData(
        GLD1, 
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
    df['GLD'] = df['close']

    return df[['date', 'GLD']]

def UVXY():
    barsList = []

    ib.reqMktData(UVXY1, '', False, False)
    ticker = ib.ticker(UVXY1)
    ib.sleep(0.1)

    sPrice = ticker.marketPrice()

    bars = ib.reqHistoricalData(
        UVXY1, 
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
    df['UVXY'] = df['close']

    return df[['date', 'UVXY']]

def SQQQ():
    barsList = []

    ib.reqMktData(SQQQ1, '', False, False)
    ticker = ib.ticker(SQQQ1)
    ib.sleep(0.1)

    sPrice = ticker.marketPrice()

    bars = ib.reqHistoricalData(
        SQQQ1, 
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
    df['SQQQ'] = df['close']

    return df[['date', 'SQQQ']]


def main():
    df = datafrm()

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

    # most likely wont need these
    # df['RSIup'] = 70
    # df['RSIdown'] = 30

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

    # Stores it into datafram
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
        if (ST[0] == None):
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

    # Merge the tickers with the main df
    GLDdf = GLD()
    df = pd.merge(df, GLDdf, on=['date'])
    UVXYdf = UVXY()
    df = pd.merge(df, UVXYdf, on=['date'])
    SQQQdf = SQQQ()
    df = pd.merge(df, SQQQdf, on=['date'])

    df.to_csv('OPdata.csv')


if __name__== '__main__':
    # put the loop logic here eto loop through everything accordingly
    main()