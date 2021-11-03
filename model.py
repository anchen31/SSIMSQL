import pandas as pd
import os
from sklearn import preprocessing
from collections import deque
import random
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import pymysql
pymysql.install_as_MySQLdb()
import mysql.connector
from mysql.connector import Error

import config

password = config.password



#pull the data and read it 

try:
    con = mysql.connector.connect(
    host = 'localhost',
    database='twitterdb', 
    user='root', 
    password = password)

    cursor = con.cursor()
    # create data db with full data set
    query = "select * from ibpy"
    cursor.execute(query)
    # get all records
    db = cursor.fetchall()

    df = pd.DataFrame(db)

except mysql.connector.Error as e:
    print("Error reading data from MySQL table", e)

    cursor.close()
    con.close()

df = df.set_axis(['date', 'open', 'high', 
                    'low', 'close', 'volume', 
                    'average', 'barCount', 'bb_bbm', 
                    'bb_bbh', 'bb_bbl', 'VWAP', 
                    'RSI', 'STsupp', 'STres', 
                    'LTsupp', 'LTres', 'GLD', 
                    'UVXY', 'SQQQ'], axis=1, inplace=False)


train_dates = df['date']

cols = list(df)[1:]

df_for_training = df[cols].astype(float)

scaler = StandardScaler()

scaler = scaler.fit(df_for_training)

df_for_training_scaled = scaler.transform(df_for_training)
print(type(df_for_training_scaled))

plt.plot(df_for_training_scaled)
plt.show()