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

# changes a timestamp to datetime format
def toDateTime(yabadabadoo):
    #toDateTime = datetime.strptime(yabadabadoo, '%Y-%m-%d %H:%M:%S')
    toDateTime = yabadabadoo.to_pydatetime()
    return toDateTime

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
    df1['tweetsent'].round(decimals = 4)

    return df1


def main():

    time.sleep(70)

    # constantly refreshes to check if there is a new ticker update
    while(True):

        now = datetime.now()

        # will add data after each minute and 5 seconds
        while now.second != 5:
            time.sleep(1)
            print(now.second)
            now = datetime.now()

        # GET THE SENTIMENT FROM THE TWITTER AND STORE IT ALL TOGETHER
        # STORE INTO MYSQL DB

        #works LOL
        df = df_resample_sizes()
        engine = create_engine(config.engine)
        with engine.begin() as connection:
            df.to_sql(name='tweetdb', con=connection, if_exists='replace', index=False)


        #Fix this and test this out
        #Change 

        try:
            con = mysql.connector.connect(
            host = 'localhost',
            database='twitterdb', 
            user='root', 
            password = password)
            print("You are connected to mySQL")
            

            if con.is_connected():
                """
                Insert twitter data
                """
                cursor = con.cursor("SELECT \
                        ibpy.date AS ibpy, \
                        tweetdb.date AS tweetdb \
                        FROM ibpy \
                        LEFT JOIN tweetdb ON ibpy.date = products.id")

                cursor.execute()
                con.commit()
                
                    
            except Error as e:
            
                print(e)

            cursor.close()
            con.close()






        # use if necessary
        #time.sleep(1)
        # df = df_resample_sizes()
        # df.plot('timestamp_ms', 'tweetsent')
        # plt.show()



if __name__== '__main__':
    main()





