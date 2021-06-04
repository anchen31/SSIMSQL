import mysql.connector

mydb = mysql.connector.connect(
    host = 'localhost',
    database='twitterdb', 
    user='root', 
    password = '@ndych3n1454L46i5Z9')

mycursor = mydb.cursor()

mycursor.execute("SELECT * FROM TwitterSent")
#mycursor.execute("TRUNCATE TABLE TwitterSent") #// deletes all the data in table
#mycursor.execute("DROP TABLE TwitterSent")
#mycursor.execute("ALTER TABLE TwitterSent ADD sentiment VARCHAR(50)")
#mycursor.execute("CREATE TABLE TwitterSent (timestamp_ms VARCHAR(20), tweet VARCHAR(255), sentiment VARCHAR(10))")

myresult = mycursor.fetchall()

for x in myresult:
  print(x)