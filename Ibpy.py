from ib_insync import *
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=2)

sia = SentimentIntensityAnalyzer()

def datetime_from_utc_to_local(utc_datetime):
    now_timestamp = time.time()
    offset = datetime.fromtimestamp(now_timestamp) - datetime.utcfromtimestamp(now_timestamp)
    return utc_datetime + offset

contract1 = Stock('SPY', 'SMART', 'USD')

#1 pull data from here

# bars = ib.reqHistoricalData(
#     contract1, 
#     endDateTime='',
#     durationStr='1 D',
#     barSizeSetting='1 min',
#     whatToShow='TRADES',
#     useRTH=True,
#     formatDate=1,
#     keepUpToDate=True)

# dt = ''
# barsList = []


# ib.reqMktData(contract1, '', False, False)
# ticker = ib.ticker(contract1)
# ib.sleep(0.1)

# sPrice = ticker.marketPrice()
# print(sPrice)

# #gets market price
# data = ib.reqMktData(contract1)
# print(data.marketPrice())
###

# while True:
#     bars = ib.reqHistoricalData(
#         contract1, 
#         endDateTime='',
#         durationStr='1 D',
#         barSizeSetting='1 min',
#         whatToShow='TRADES',
#         useRTH=True,
#         formatDate=1,
#         keepUpToDate=True)
#     if not bars:
#         break
#     barsList.append(bars)

# allBars = [b for bars in reversed(barsList) for b in bars]
# df = util.df(allBars)

# print(df)

#i can just do the df into a sql thingy again lol






plt.rcParams['figure.figsize'] = [12, 7]
plt.rc('font', size=14)

name = 'SPY'
ticker = yfinance.Ticker(name)
df1 = ticker.history(interval="5m",start="2021-08-12",end="2021-08-13", threads= False)
df1['Date'] = pd.to_datetime(df1.index)
df1['Date'] = df1['Date'].apply(mpl_dates.date2num)
df1 = df1.loc[:,['Date', 'Open', 'High', 'Low', 'Close']]




df = pd.DataFrame(columns=['date', 'sentiment'])

h = 0
newsProviders = ib.reqNewsProviders()
codes = '+'.join(np.code for np in newsProviders)

ib.qualifyContracts(contract1)
# headlines = ib.reqHistoricalNews(contract1.conId, codes, '2021-08-12', '2021-08-13', 50)
headlines = ib.reqHistoricalNews(contract1.conId, codes, '', '', 50)

for i in headlines:
    latest = i.headline
    # time = i.time
    # time = datetime_from_utc_to_local(time)
    vs = sia.polarity_scores(latest)
    sentiment = round(vs['compound'], 4)
    time = i.time
    print(time, sentiment, latest)
    df.loc[h] = [time, sentiment]

    h = h+1



fig, ax = plt.subplots()
candlestick_ohlc(ax,df1.values,width=0.0001, \
               colorup='green', colordown='red', alpha=0.8)
date_format = mpl_dates.DateFormatter('%Y-%m-%d %H:%M:%S')
ax.xaxis.set_major_formatter(date_format)
fig.autofmt_xdate()
fig.tight_layout()



df['sentiment'] = df['sentiment'].rolling(int(len(df)/5)).mean()
df.plot('date', 'sentiment')


plt.show()


# article = ib.reqNewsArticle(latest.providerCode, latest.articleId)
# print(article)












ib.disconnect()



