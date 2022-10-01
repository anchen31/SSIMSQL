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
# import talib as ta
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
# from darts.models import TransformerModel
from darts.metrics import mape, rmse
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression
from sklearn.linear_model import LinearRegression
import requests, csv, pytz
import json

# can work across multiple time frames 
BASE_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
START_DATE = '2016-01-25'

r = requests.get("{}/{}".format(BASE_URL, START_DATE))
data = r.json()
print(json.dumps(data, indent=2))


fg_data = data['fear_and_greed_historical']['data']

# fg_date = fg_data.to_csv()

print(fg_data)

# fear_greed_values = {}

# FEAR_GREED_CSV_FILENAME = 'datasets/fear-greed-2021-2022.csv'

# with open(FEAR_GREED_CSV_FILENAME, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['Date,,,,Fear Greed'])

#     for data in fg_data:
#         dt = datetime.fromtimestamp(data['x'] / 1000, tz=pytz.utc)
#         fear_greed_values[dt.date()] = int(data['y'])
#         writer.writerow([dt.date(), "", "", "", int(data['y'])])


