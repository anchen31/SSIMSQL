import mysql.connector

mydb = mysql.connector.connect(
    host = 'localhost',
    database='twitterdb', 
    user='root', 
    password = '@ndych3n1454L46i5Z9')

mycursor = mydb.cursor(buffered=True)

mycursor.execute("SELECT * FROM TwitterSent")
#mycursor.execute("TRUNCATE TABLE TwitterSent") 
#// deletes all the data in table
#mycursor.execute("DROP TABLE TwitterSent")
#mycursor.execute("ALTER TABLE TwitterSent ADD sentiment VARCHAR(50)")
#mycursor.execute("CREATE TABLE TwitterSent (timestamp_ms VARCHAR(20), sentiment VARCHAR(10))")

#mycursor.execute("CREATE TABLE rednewsDB (timestamp_ms VARCHAR(20), reddit_sentiment VARCHAR(10), reddit_comm_sentiment VARCHAR(10), news_sentiment VARCHAR(10))")
#mycursor.execute("DROP TABLE rednewsDB")
#mycursor.execute("TRUNCATE TABLE rednewsDB")
#mycursor.execute("SELECT * FROM rednewsDB")

myresult = mycursor.fetchall()

for x in myresult:
  print(x)