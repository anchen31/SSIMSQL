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


barSze = '1 hour'
durStrng = '8 M'



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

# does all of the ta stuff and puts it into a mysql db
def main():

    # what  do i need to test?
    # loop it and have a time limit to kill it and disconnect from ib
    # add a time sleep mode that catches 
    # each time it makes 
    counter = 0

    while True:
        now = datetime.now()

        #pulls data after each minute and
        while now.second != 1:
            time.sleep(1)
            print(now.second)
            now = datetime.now()

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

        #print(df.columns)

        print(df.tail())
        
        #################################################Create a new db for this data, this will be the main db that will have everything else join it###
        engine = create_engine(config.engine)
        ############################## Create config.engine1 that has a different db loaction #######################
        with engine.begin() as connection:
            #maybe "replace" fixes the missing values problem?
            df.to_sql(name='ibpy', con=connection, if_exists='replace', index=False)

    ib.disconnect()

#df = dropna(df)


# i can just do the df into a sql thingy again lol



if __name__== '__main__':
    # put the loop logic here eto loop through everything accordingly
    main()


#################################################################### WHY IS BENZINGA DATA 250$ LOL #####################################################################

# plt.rcParams['figure.figsize'] = [12, 7]
# plt.rc('font', size=14)

# name = 'SPY'
# ticker = yfinance.Ticker(name)
# df1 = ticker.history(interval="5m",start="2021-08-12",end="2021-08-13", threads= False)
# df1['Date'] = pd.to_datetime(df1.index)
# df1['Date'] = df1['Date'].apply(mpl_dates.date2num)
# df1 = df1.loc[:,['Date', 'Open', 'High', 'Low', 'Close']]




# df = pd.DataFrame(columns=['date', 'sentiment'])

# h = 0
# newsProviders = ib.reqNewsProviders()
# codes = '+'.join(np.code for np in newsProviders)

# print(codes)

# ib.qualifyContracts(contract1)

# td = datetime.today()
# td1 = td.strftime("%Y-%m-%d")

# print(td1)

# tmr = td.strftime("%Y-%m-%d+1")

# print(tmr)

# # the number parameter should stay at 50
# headlines = ib.reqHistoricalNews(contract1.conId, codes, '2021-09-19', '2021-09', 50)

# #if the day doesn't match up then you remove the headline
# # store headlines into a manipulatable list
# hList = []

# today = datetime.today()
# today = today.day

# for j in headlines:
#     day = j.time
#     day = day.day
#     if day == today:
#         hList.append(j)

# for i in headlines:
#     print(i)


# for i in hList:
#     latest = i.headline
#     # turn headline into pst time and remove the seconds into 0
#     # next, 
#     time = i.time
#     time = time.replace(minute = 0)
#     print(time)

#     #datetime_from_utc_to_local(time)
#     vs = sia.polarity_scores(latest)
#     sentiment = round(vs['compound'], 4)
#     time = i.time
#     #print(time, sentiment, latest)


#     df.loc[h] = [time, sentiment]

#     h = h+1



# fig, ax = plt.subplots()
# candlestick_ohlc(ax,df1.values,width=0.0001, \
#                colorup='green', colordown='red', alpha=0.8)
# date_format = mpl_dates.DateFormatter('%Y-%m-%d %H:%M:%S')
# ax.xaxis.set_major_formatter(date_format)
# fig.autofmt_xdate()
# fig.tight_layout()



# df['sentiment'] = df['sentiment'].rolling(int(len(df)/5)).mean()
# df.plot('date', 'sentiment')


# plt.show()


# article = ib.reqNewsArticle(latest.providerCode, latest.articleId)
# print(article)




