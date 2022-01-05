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


# gets the raw twitter data from TSSmySQL and organizes it into one minute chunks and stores it in twitterdb
# it also pulls data from ibpy db and merges twitter db and ibpy db together into ibpy db.
# basically creates super db

def ibpyData():
    try:
        con = mysql.connector.connect(
        host = 'localhost',
        database='twitterdb', 
        user='root', 
        password = password)

        cursor = con.cursor()
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
    return df



# returns a df of the twitter data organized by the minute
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
    holder = datetime.strptime(holder[0], '%Y-%m-%d %H:%M:%S')
    holder = holder.minute
    counter = 0
    total = 0


    df1 = pd.DataFrame(columns = ['date', 'tweetsent'])

    for index, row in df.iterrows():

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
            df1 = df1.append({'date':date1, 'tweetsent':total}, ignore_index=True)
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
    df1['tweetsent'].round(decimals = 4)

    return df1


def main():

    time.sleep(70)

    # constantly refreshes to check if there is a new ticker update
    while(True):

        now = datetime.now()

        # will add data after each minute and 4 seconds
        while now.second != 4:
            time.sleep(1)
            print(now.second)
            now = datetime.now()

        # Stores data onto mysql
        # works LOL
        df = df_resample_sizes()
        engine = create_engine(config.engine)
        with engine.begin() as connection:
            df.to_sql(name='tweetdb', con=connection, if_exists='replace', index=False)

        # pull out ibpydata into a df
        df1 = ibpyData()
        # merge twitter and ibpy data
        result = pd.merge(df,df1, on='date', how='left')

        ### Get result and do a correlation plot with ith and see
        


        ############################################### TEST THIS PART OUT LATER
        # add into another db on mysql
        engine = create_engine(config.engine)

        # Create a new db to store result in
        with engine.begin() as connection:
            # rename ibpy to something else
            result.to_sql(name='ibpy', con=connection, if_exists='replace', index=False)
        # store into new data thing





if __name__== '__main__':
    main()





