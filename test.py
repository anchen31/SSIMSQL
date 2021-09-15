import mysql.connector

mydb = mysql.connector.connect(
    host = 'localhost',
    database='twitterdb', 
    user='root', 
    password = '@ndych3n1454L46i5Z9')

mycursor = mydb.cursor(buffered=True)


#mycursor.execute("SELECT * FROM TwitterSent")
#mycursor.execute("SELECT * FROM tweetdb")
#mycursor.execute("TRUNCATE TABLE TwitterSent") 
#// deletes all the data in table
#mycursor.execute("DROP TABLE TwitterSent")
#mycursor.execute("ALTER TABLE TwitterSent ADD sentiment VARCHAR(50)")
#mycursor.execute("CREATE TABLE TwitterSent (timestamp_ms VARCHAR(20), sentiment VARCHAR(10))")

#mycursor.execute("CREATE TABLE rednewsDB (timestamp_ms VARCHAR(20), reddit_sentiment VARCHAR(10), reddit_comm_sentiment VARCHAR(10), news_sentiment VARCHAR(10))")
#mycursor.execute("DROP TABLE rednewsDB")
#mycursor.execute("TRUNCATE TABLE rednewsDB")
#mycursor.execute("SELECT * FROM rednewsDB")

mycursor.execute("CREATE TABLE IBPY (date VARCHAR(20), open VARCHAR(10), high VARCHAR(10), low VARCHAR(10), \
  close VARCHAR(10), volume VARCHAR(10), average VARCHAR(10), barCount VARCHAR(10), bb_bbm VARCHAR(10), bb_bbh VARCHAR(10), \
  bb_bbl VARCHAR(10), VWAP VARCHAR(10), RSI VARCHAR(10), STsupp VARCHAR(10), STres VARCHAR(10), LTsupp VARCHAR(10), LTres VARCHAR(10), \
  GLD VARCHAR(10), UVXY VARCHAR(10), SQQQ VARCHAR(10))")
#mycursor.execute("DROP TABLE IBPY")
#mycursor.execute("TRUNCATE TABLE IBPY")
#mycursor.execute("SELECT * FROM IBPY")


#myresult = mycursor.fetchall()

#for x in myresult:
#  print(x)