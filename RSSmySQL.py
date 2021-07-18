import praw
import nltk
import time
import pandas as pd
from datetime import datetime
import mysql.connector
from mysql.connector import Error
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import matplotlib.pyplot as plt

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

# changes a string to datetime format
def toDateTime(yabadabadoo):
    toDateTime = datetime.strptime(yabadabadoo, '%Y-%m-%d %H:%M:%S')
    return toDateTime


################################################### change to connect to the main df from ib
# gets the time from the main database 
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

# will get scores of each post
def getRedditSentiment():
    scoreList = []
    score = 0
    submissions = 0

    # change this so that the ticker can be changed externally 
    # chooses the top posts 
    hot_posts = reddit.subreddit(config.stock).hot()
    # will get reddit and comment score

    for submission in hot_posts:
        # will store in vs and append it all into a list and compact it
        vs = sia.polarity_scores(submission.title)
        sentiment = vs['compound']

        # adds onto the score list
        scoreList.append(sentiment)
        submissions += 1
        

    for x in scoreList:
        score += x

    score = score/submissions

    return score



def getNewSentiment():
    score = 0
    return score


def getMarketSentiment():
    score = 0
    return score


# returns combined sentiment score from twitter db
def getTweetSentiment():
    score = 0
    holder = -1
    run = True
    scoreL = []
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

        minCompare = toDateTime(df[0].iloc[holder])
        minCompare = minCompare.minute


        while(run):
            Compare = toDateTime(df[0].iloc[holder-1])
            Compare = Compare.minute

            if minCompare == Compare:
                # append it to the holder list
                scoreL.append(float(df[1].iloc[holder]))
                holder -= 1
            else:
                scoreL.append(float(df[1].iloc[holder]))
                # calulate the score and return it 
                holder = 0
                for x in scoreL:
                    holder += 1
                    score += x

                score = score/holder
                #print("final score", score)

                run = False
                return score



    except mysql.connector.Error as e:
        print("Error reading data from MySQL table", e)

    cursor.close()
    con.close()


    return score

# returns a df of the twitter data organized by the minute
def df_resample_sizes(df):
    Holder_List = []
    holder = df[0]
    holder = toDateTime(holder[0])
    holder = holder.minute
    counter = 0
    total = 0

    df1 = pd.DataFrame(columns = [0, 1])

    for index, row in df.iterrows():
        date1 = toDateTime(row[0])
        date = date1.minute
        if(date != holder):
            holder = date
            total = sum(Holder_List)
            try:
                total = total/counter  # will have to use later after i implement rounding on tssmysql     
                total = round(total, 4)
            except ZeroDivisionError as e:
                pass

            df1 = df1.append({0:date1, 1:total}, ignore_index=True)
            #print(date, total) #shows the condensed data organized

            #resets the params
            total = 0
            Holder_List = []
            counter = 0

        else:
            Holder_List.append(float(row[1]))
            counter = counter + 1

    return df1


def main():
    run = True
    timeout = time.time() + 1
    timeCompare = getTime()
    timeNow = toDateTime(timeCompare[0])

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

    except mysql.connector.Error as e:
        print("Error reading data from MySQL table", e)

    cursor.close()
    con.close()

    df = df_resample_sizes(df)
    df[1] = df[1].rolling(int(len(df)/5)).mean()

    #print(df)

    x = df[0]
    y = df[1]

    plt.plot(x,y)

    plt.show()

    print("dont read lmaoo")

    ################################################

    # constantly refreshes to check if there is a new ticker update
    while(run):

        timeCompare = getTime()
        timePast = toDateTime(timeCompare[1])

        # whenever we detect that interactive brokers data has updated into the mysql database
        # we will add to our table and then join to it
        # IF THERE IS A NEW TICKER
        if timePast == timeNow:
            #add
            timeNow = toDateTime(timeCompare[0])
            # GET THE SENTIMENT FROM THE TWITTER AND STORE IT ALL TOGETHER
            # STORE INTO MYSQL DB

            # use to connect method to store into the db
            # (timestamp, twitter, reddit, reddit comment?, news on stock from finviz, overall stock news s&p500, s&p500 data?)
            #connect(timeNow, )
            # then join it to the main table







        # while the ticker hasn't refreshed, we will keep gathering the latest sentiment
        else:
            #gather reddit sentiment score

            #gather reddit comment score


            #should I put this here?
            #don't change the news score if there isn't any new news, base it off of the most recent news
            #gather news about overall market??

            #gather news about overall stock??

            #getTweetSentiment() #works
            #print(getRedditSentiment()) #works
            print(timeNow)
            print(timePast)





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





