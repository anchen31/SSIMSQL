from ib_insync import *
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import pytz
import pandas as pd
import numpy as np
import yfinance
from ta.utils import dropna
from ta.volatility import BollingerBands
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=2)

sia = SentimentIntensityAnalyzer()

contract1 = Stock('SPY', 'SMART', 'USD')

def isFarFromLevel(l):
    return np.sum([abs(l-x) < s  for x in levels]) == 0

def isSupport(df,i):
    support = df['low'][i] < df['low'][i-1]  and df['low'][i] < df['low'][i+1] and df['low'][i+1] < df['low'][i+2] and df['low'][i-1] < df['low'][i-2]
    return support


def isResistance(df,i):
    resistance = df['high'][i] > df['high'][i-1]  and df['high'][i] > df['high'][i+1] and df['high'][i+1] > df['high'][i+2] and df['high'][i-1] > df['high'][i-2]
    return resistance

def closest(lst, K):
    lst.sort()
    
    return 

# 1 pull data from here

bars = ib.reqHistoricalData(
    contract1, 
    endDateTime='',
    durationStr='1 D',
    barSizeSetting='1 min',
    whatToShow='TRADES',
    useRTH=True,
    formatDate=1,
    keepUpToDate=True)

dt = ''
barsList = []


ib.reqMktData(contract1, '', False, False)
ticker = ib.ticker(contract1)
ib.sleep(0.1)

sPrice = ticker.marketPrice()

while True:
    bars = ib.reqHistoricalData(
        contract1, 
        endDateTime='',
        durationStr='1 D',
        barSizeSetting='1 min',
        whatToShow='TRADES',
        useRTH=True,
        formatDate=1,
        keepUpToDate=True)
    if not bars:
        break
    barsList.append(bars)

allBars = [b for bars in reversed(barsList) for b in bars]
df = util.df(allBars)




#TA STUFF HERE ######################################

indicator_bb = BollingerBands(close=df["close"], window=20, window_dev=2)

df['bb_bbm'] = indicator_bb.bollinger_mavg()
df['bb_bbh'] = indicator_bb.bollinger_hband()
df['bb_bbl'] = indicator_bb.bollinger_lband()

v = df['volume']
p = df['close']

df['VWAP'] = ((v * p).cumsum() / v.cumsum())

# LT AND ST S/R ######################################

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



#df = dropna(df)


# i can just do the df into a sql thingy again lol






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


ib.disconnect()



