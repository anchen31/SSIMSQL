import praw
import nltk
import time
import time
import pandas as pd
from datetime import datetime
from textblob import TextBlob
import mysql.connector
from mysql.connector import Error
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import config

sia = SentimentIntensityAnalyzer()
#connect to reddit client

reddit = praw.Reddit(client_id=config.client_id,
                     client_secret=config.client_secret,
                     user_agent=config.user_agent)


password = config.password


#connect to mysql method to add data
def connect(timestamp_ms, reddit_sentiment, reddit_comm_sentiment, news_sentiment):
    """
    connect to MySQL database and insert twitter data
    """
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
            cursor = con.cursor()
            # twitter
            query = "INSERT INTO rednewsDB (timestamp_ms, reddit_sentiment, reddit_comm_sentiment, news_sentiment) VALUES (%s, %s, %s, %s)"
            cursor.execute(query, (timestamp_ms, reddit_sentiment, reddit_comm_sentiment, news_sentiment))
            con.commit()
            
            
    except Error as e:
    
        print(e)

    cursor.close()
    con.close()

    return

#gets the time from the main database 
def getTime():
    date = []
    try:
        con = mysql.connector.connect(
        host = 'localhost',
        database='twitterdb', 
        user='root', 
        password = config.password)

        cursor = con.cursor()
        query = "select * from TwitterSent"
        cursor.execute(query)
        # get all records
        db = cursor.fetchall()

        df = pd.DataFrame(db)

        date.append(df[0].iloc[-1])
        date.append(df[0].iloc[-2])


    except mysql.connector.Error as e:
        print("Error reading data from MySQL table", e)

    cursor.close()
    con.close()

    return date


def getRedditSentiment():
    score = []
    # will get reddit and comment score
    return score



def getNewSentiment():
    score = 0
    return score


def getMarketSentiment():

def main():
    run = True
    timeout = time.time() + 20
    timeCompare = getTime()
    timeNow = datetime.strptime(timeCompare[0], '%Y-%m-%d %H:%M:%S')

    while(run):

        timeCompare = getTime()
        timePast = datetime.strptime(timeCompare[1], '%Y-%m-%d %H:%M:%S')

        # whenever we detect that interactive brokers data has updated into the mysql database
        # we will add to our table and then join to it
        # IF THERE IS A NEW TICKER
        if timePast == timeNow:
            #add
            timeNow = datetime.strptime(timeCompare[0], '%Y-%m-%d %H:%M:%S')

            #connect to the twitter db

            # GET THE SENTIMENT FROM THE TWITTER AND STORE IT ALL TOGETHER

            # use to connect method to store into the db?
            # then join it to the main table





        # while its not the time, we will keep gathering the latest sentiment
        else:
            #gather reddit sentiment score

            #gather reddit comment score


            #should I put this here?
            #don't change the news score if there isn't any new news, base it off of the most recent news
            #gather news about overall market??

            #gather news about overall stock??










        # use if necessary
        #time.sleep(1)








        # kills main after a certain amount of time
        test = 0
        if test == 5 or time.time() > timeout:
            run = False
            break
        test = test - 1




if __name__== '__main__':
    main()





