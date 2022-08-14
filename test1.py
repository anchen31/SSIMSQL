import pandas as pd
import seaborn as sns
import numpy as np
import datetime as dt 
import yfinance
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
import talib as ta

plt.rcParams['figure.figsize'] = [12, 7]
plt.rc('font', size=14)

name = 'SPY'
ticker = yfinance.Ticker(name)
df = ticker.history(interval="1d",start="2004-1-15", end="2022-05-15")
df['Date'] = pd.to_datetime(df.index)
df['Date'] = df['Date'].apply(mpl_dates.date2num)
df = df.loc[:,['Date', 'Open', 'High', 'Low', 'Close']]

# df = pd.read_csv('data.csv')
# df = df.loc[:,['date', 'open', 'high', 'low', 'close']]
# df['date'] = pd.to_datetime(df.index)
# # df = df.loc[:,['open', 'high', 'low', 'close']]
# print(type(df['open'].values))
# print(df)

def isSupport(df,i):
    support = df['Low'][i] < df['Low'][i-1]  and df['Low'][i] < df['Low'][i+1] and df['Low'][i+1] < df['Low'][i+2] and df['Low'][i-1] < df['Low'][i-2]
    return support
def isResistance(df,i):
    resistance = df['High'][i] > df['High'][i-1]  and df['High'][i] > df['High'][i+1] and df['High'][i+1] > df['High'][i+2] and df['High'][i-1] > df['High'][i-2]
    return resistance 

data = []
for i in range(2,df.shape[0]-2):
  if isSupport(df,i):
    data.append((df['Date'][i],df['Low'][i], 1))
  elif isResistance(df,i):
    data.append((df['Date'][i],df['High'][i], -1))


#print out levels and see lows and high and assign them a triangle


def plot_all():
  fig, ax = plt.subplots()
  candlestick_ohlc(ax,df.values,width=0.08, \
                   colorup='green', colordown='red', alpha=0.8)
  date_format = mpl_dates.DateFormatter('%d %b %Y')
  ax.xaxis.set_major_formatter(date_format)
  fig.autofmt_xdate()
  fig.tight_layout()
  for level in data:
    if level[2] == 1:
      plt.plot(level[0], level[1], marker='^', color='green')
    elif level[2] == -1:
      plt.plot(level[0], level[1], marker='v', color='red')

  plt.show()

def get_macd(price, slow, fast, smooth):
    exp1 = price.ewm(span = fast, adjust = False).mean()
    exp2 = price.ewm(span = slow, adjust = False).mean()
    macd = pd.DataFrame(exp1 - exp2)
    signal = pd.DataFrame(macd.ewm(span = smooth, adjust = False).mean())
    hist = macd - signal
    return hist

# takes a df and adds features to it
def add_features(data):
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
  data['count'] = range(len(data))

  return data


# plot_all()

# high is 1 and low is 

df1 = pd.DataFrame(data, columns=['Date', 'price', 'sup_res'])

# get regular df and add columns to it
df = add_features(df)

# merge the two dfs together
merged_df = pd.merge(df1, df, on='Date')

# print(df1)
# print(df)
print(merged_df)

# seperate the -1 and positive 1 values and graph data
sell = merged_df.loc[merged_df['sup_res'] == -1]

buy = merged_df.loc[merged_df['sup_res'] == 1]




















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


# ########################################################### heat map below #####
# with 4 year data, it is more responsive to the indexes such as gld 
# with the intrad day data, it is less efficent

# import pandas as pd
# import seaborn as sns

df = pd.read_csv('four_year_date.csv', index_col=False)

# # df = pd.read_csv('OPdata.csv', index_col=False)
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

# ##############################################################

# uvxy, rio,? uso?, sqqq

# Naw = ['date', 'open', 'high', 'low', 'volume', 'average', 'barCount', 'bb_bbm', 'bb_bbh', 'bb_bbl', 'VWAP', 'RSI', 'STsupp', 'STres', 'LTsupp', 'LTres']
# Naw = ['date']

# df = df.dropna()

# # orignal data look
# del df['volume']
# del df['barCount']
# del df['UVXY']
# del df['SQQQ']

# df.plot()
# plt.show()


# # # print(df['LTres']. head(50))

# for i in df.columns:
#   if i in Naw:
#     pass
#   else:
#     # does the linear regression on the columns
#     X = df.index.values
#     y = df[[i]].values

#     length = len(X)

#     X = X.reshape(length, 1)
#     y = y.reshape(length, 1)

#     regressor = LinearRegression()
#     regressor.fit(X, y)

#     y_pred1 = regressor.predict(X)
#     y_pred = pd.DataFrame(y_pred1)

#     # create a new value based off of 
#     df[i] = df[i] - y_pred[0]
#     # print(y_pred[0])

#     # plotting
#     plt.scatter(X, y, color="black")
#     plt.plot(X, y_pred1, color="blue", linewidth=3)

#     plt.xticks(())
#     plt.yticks(())
#     plt.title(i)
#     plt.show()



# # del df['volume']
# # del df['barCount']
# # del df['UVXY']
# # del df['SQQQ']
# # del df['LTres']

# df.plot()
# plt.show()


# # plt.figure(figsize = (15,15))
# # sns.set(font_scale=0.75)
# # ax = sns.heatmap(df.corr().round(3), 
# #             annot=True, 
# #             square=True, 
# #             linewidths=.75, cmap="coolwarm", 
# #             fmt = ".2f", 
# #             annot_kws = {"size": 11})
# # ax.xaxis.tick_bottom()
# # plt.title("correlation matrix")
# # plt.show()

# ###################################### normalize the trade chart











