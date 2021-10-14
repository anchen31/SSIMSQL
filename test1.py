import time
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine
import pymysql
pymysql.install_as_MySQLdb()
import mysql.connector
from mysql.connector import Error

import matplotlib.pyplot as plt

import config

password = config.password


def df_resample_sizes():
    try:
        con = mysql.connector.connect(
        host = 'localhost',
        database='twitterdb', 
        user='root', 
        password = password)

        cursor = con.cursor()
        query = "select * from TwitterSent"
        cursor.execute(query)
        # get all records
        db = cursor.fetchall()

        df = pd.DataFrame(db)

    except mysql.connector.Error as e:
        print("Error reading data from MySQL table", e)

    cursor.close()
    con.close()

    Holder_List = []
    holder = df[0]
    #holder = toDateTime(holder[0])
    holder = datetime.strptime(holder[0], '%Y-%m-%d %H:%M:%S')
    holder = holder.minute
    counter = 0
    total = 0


    df1 = pd.DataFrame(columns = ['timestamp_ms', 'tweetsent'])

    for index, row in df.iterrows():
        #date1 = toDateTime(row[0])
        date1 = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
        date = date1.minute
        if(date != holder):
            holder = date
            total = sum(Holder_List)
            try:
                total = total/counter  # will have to use later after i implement rounding on tssmysql     
                total = round(total, 4)
            except ZeroDivisionError as e:
                pass


            date1 = date1.replace(second=0)
            df1 = df1.append({'timestamp_ms':date1, 'tweetsent':total}, ignore_index=True)
            #print(date, total) #shows the condensed data organized

            #resets the params
            total = 0
            Holder_List = []
            counter = 0

        else:
            Holder_List.append(float(row[1]))
            counter = counter + 1

    #I would have to round this
    df1['tweetsent'] = df1['tweetsent'].rolling(int(len(df1)/5)).mean()

    df1['tweetsent'] = df1['tweetsent'].round(decimals = 4)

    return df1['tweetsent']


print(df_resample_sizes())