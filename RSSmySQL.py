import praw
import nltk
import time
import pandas as pd
from datetime import datetime
from textblob import TextBlob
import mysql.connector
from mysql.connector import Error
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()
#connect to reddit client
# reddit = praw.Reddit(client_id='',
#                      client_secret='',
#                      user_agent='')

# password = ""


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
    try:
        con = mysql.connector.connect(
        host = 'localhost',
        database='twitterdb', 
        user='root', 
        password = password)
        print("You are connected to mySQL")

        cursor = con.cursor()
        query = "select * from TwitterSent"
        cursor.execute(query)
        # get all records
        db = cursor.fetchall()

        df = pd.DataFrame(db)

        date = df[0].iloc[-1]


    except mysql.connector.Error as e:
        print("Error reading data from MySQL table", e)

    cursor.close()
    con.close()

    return date


if __name__== '__main__':
    run = True

    while(run):
        time = datetime.now()
        time = time.strftime('%Y-%m-%d %H:%M:%S')
        time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')

        #if the time is on the database, then add onto it.
        #will make this a lil earlier

        if (time == getTime()):
            print("LOL AHAHAH")
        else:
            print("fuck")

        run = False



