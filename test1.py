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

# plt.rcParams['figure.figsize'] = [12, 7]
# plt.rc('font', size=14)

# name = 'SPY'
# ticker = yfinance.Ticker(name)
# df = ticker.history(interval="1d",start="2017-1-15", end="2022-05-15")
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


# ########################################################### heat map below #####
# with 4 year data, it is more responsive to the indexes such as gld 
# with the intrad day data, it is less efficent

# import pandas as pd
# import seaborn as sns

# df = pd.read_csv('four_year_date.csv', index_col=False)

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

# # uvxy, rio,? uso?, sqqq

# # Naw = ['date', 'open', 'high', 'low', 'volume', 'average', 'barCount', 'bb_bbm', 'bb_bbh', 'bb_bbl', 'VWAP', 'RSI', 'STsupp', 'STres', 'LTsupp', 'LTres']
# Naw = ['date']

# df = df.dropna()

# # orignal data look
# del df['volume']
# del df['barCount']
# del df['UVXY']
# del df['SQQQ']

# df.plot()
# plt.show()


# # print(df['LTres']. head(50))

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

###################################### normalize the trade chart

df = pd.read_csv('stolen_vehicles_merged.csv')

df = df.drop(['VehicleType', 'VEHICLE_YEAR', 'MAKE', 'BASIC_COLOUR', 'VEHICLE_TYPE', 'Unnamed: 0'], axis=1)

region_pops = pd.DataFrame(data={'CANTERBURY' : [578290], 
                                  'AUCKLAND CITY' : [493990], 
                                  'COUNTIES/MANUKAU': [578650], 
                                  'BAY OF PLENTY' : [259090], 
                                  'WELLINGTON' : [525910], 
                                  'WAITEMATA' : [628600], 
                                  'WAIKATO' : [435690], 
                                  'CENTRAL' : [378965], 
                                  'NORTHLAND' : [193170], 
                                  'EASTERN' : [225865], 
                                  'SOUTHERN' : [191919] })

df['COUNT'] = df_fleet['MODEL'].value_counts().values

################################### new graph ##################################


plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

f, axes = plt.subplots(1, 2)


sns.barplot(x=df["COUNT"], y=df['VehicleModel'], data=df, ax=axes[0]).set(xlabel="COUNT", ylabel = "Vehicle Model", title= 'n')
sns.barplot(x=df["COUNT"], y=df['ORIGINAL_COUNTRY'], data=df, ax=axes[1]).set(xlabel="COUNT", ylabel = "Original Country", title= 'n')

plt.show()



################################### new graph Cars Stolen/Location Population##################################

df1 = region_pops.iloc[0].to_frame()

df2 = df['Location'].value_counts().to_frame()

# merge them together
data = pd.merge(df1, df2, left_index=True, right_index=True).reset_index()

data['average'] = data['Location']/data[0]

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

f, axes = plt.subplots(1, 3)

sns.set(font_scale = 0.8)

sns.barplot(y="index", x="Location", data=data, ax=axes[0]).set(xlabel="Cars Stolen", ylabel = "Location", title= 'number of cars stolen per district')
sns.barplot(y="index", x=data[0], data=data, ax=axes[1]).set(xlabel="Population", ylabel = None, yticks=[], title= 'population per district')
sns.barplot(y="index", x=data['average'], data=data, ax=axes[2]).set(xlabel="Cars Stolen/Location Population", ylabel = None, yticks=[], title= 'proportion of cars stolen to population size per district')

plt.show()



################################### Month chart ##################################

df['DateStolen'] = pd.to_datetime(df['DateStolen'], format = '%d/%m/%Y').dt.to_period('M')

df = df.sort_values(by='DateStolen',ascending=False)

data = df['DateStolen'].value_counts().to_frame()

month_graph = sns.barplot(x = data.index, y = 'DateStolen',data=data, palette='hls')

month_graph.set(xlabel ="Month Stolen", ylabel = "# Cars Stolen", title ='Cars Stolen Count by Month')

plt.show()

################################### replacement for one of them ##################################

# sets the color pallete
colors = sns.color_palette('pastel')[0:5]
# gets the value counts and puts it into order for easy manipulation
order = df['ORIGINAL_COUNTRY'].value_counts()
# turns the series into a dataframe
order = order.to_frame()
# because there are a lot of contries that equal to less than 0 percent of the total amount, we removed them
order = order[order['ORIGINAL_COUNTRY'] >= 30]
# sets the value for the pie chart
data = order.iloc[:, 0]
labels = order.index
# plots pie chart
pie = plt.pie(data, colors = colors, autopct='%.0f%%', pctdistance = 1.1)
plt.legend(pie[0], labels, loc="best")

# plt.show()



  














