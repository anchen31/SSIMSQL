import pandas as pd
import seaborn as sns
import numpy as np
import yfinance
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

# plt.rcParams['figure.figsize'] = [12, 7]
# plt.rc('font', size=14)

# name = 'SPY'
# ticker = yfinance.Ticker(name)
# df = ticker.history(interval="1d",start="2019-1-15", end="2020-07-15")
# df['Date'] = pd.to_datetime(df.index)
# df['Date'] = df['Date'].apply(mpl_dates.date2num)
# df = df.loc[:,['Date', 'Open', 'High', 'Low', 'Close']]

# # df = pd.read_csv('data.csv')
# # df = df.loc[:,['date', 'open', 'high', 'low', 'close']]
# # df['date'] = pd.to_datetime(df.index)
# # # df = df.loc[:,['open', 'high', 'low', 'close']]
# # print(type(df['open'].values))
# print(df)

# def isSupport(df,i):
#     support = df['Low'][i] < df['Low'][i-1]  and df['Low'][i] < df['Low'][i+1] and df['Low'][i+1] < df['Low'][i+2] and df['Low'][i-1] < df['Low'][i-2]
#     return support
# def isResistance(df,i):
#     resistance = df['High'][i] > df['High'][i-1]  and df['High'][i] > df['High'][i+1] and df['High'][i+1] > df['High'][i+2] and df['High'][i-1] > df['High'][i-2]
#     return resistance 

# data = []
# for i in range(2,df.shape[0]-2):
#   if isSupport(df,i):
#     data.append((df['Date'][i],df['Low'][i], 1))
#   elif isResistance(df,i):
#     data.append((df['Date'][i],df['High'][i], -1))


# #print out levels and see lows and high and assign them a triangle


# def plot_all():
#   fig, ax = plt.subplots()
#   candlestick_ohlc(ax,df.values,width=0.08, \
#                    colorup='green', colordown='red', alpha=0.8)
#   date_format = mpl_dates.DateFormatter('%d %b %Y')
#   ax.xaxis.set_major_formatter(date_format)
#   fig.autofmt_xdate()
#   fig.tight_layout()
#   for level in data:
#     if level[2] == 1:
#       plt.plot(level[0], level[1], marker='^', color='green')
#     elif level[2] == -1:
#       plt.plot(level[0], level[1], marker='v', color='red')

#   plt.show()


# plot_all()
# total = 0
# total1 = 0
# count = 0
# count1 = 0

# for stuff in data:
#   if stuff[2] == 1:
#     total -= stuff[1]
#     count += 1

#     while count1 > 0:
#       total -= stuff[1]
#       count1 -= 1

#   if stuff[2] == -1:
#     while count > 0:
#       total += stuff[1]
#       count -= 1

#     total += stuff[1]
#     count1 += 1



# print(total)

# plot_all()


########################################################### heat map below #####
# with 4 year data, it is more responsive to the indexes such as gld 
# with the intrad day data, it is less efficent

<<<<<<< HEAD
import pandas as pd
import seaborn as sns

df = pd.read_csv('four_year_date.csv', index_col=False)

# df = pd.read_csv('OPdata.csv', index_col=False)
# # df = pd.read_csv('OPdata.csv', index_col=False)
# # df = df.drop('Unnamed: 0',1)
# df = df.drop('date', 1)

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



##############################################################

df = pd.read_csv('OPdata.csv', index_col=0)

min_max_scaler = preprocessing.MinMaxScaler()
# scaler = MinMaxScaler(feature_range = (0,1))
scaler = StandardScaler()




df = scaler.fit_transform(df)

df = pd.DataFrame(df)

df.plot()
plt.show()




