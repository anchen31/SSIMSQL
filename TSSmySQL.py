import mysql.connector
from mysql.connector import Error
import tweepy
import json
from dateutil import parser
import time
import os
import subprocess

#importing file which sets env variable
#subprocess.call("./settings.sh", shell = True)


consumer_key = 'd9iECNcgQsRoNQXkFsikk8dsC'
consumer_secret = 'hA48EPaFwl3vdmfPyYDnOs0qQnr8q3aL1FtTvoQvBoabkbS7S8'
access_token = '1291554276884406272-ucfXmUs5fkiRgG6Eqzah3TnMoPHKEi'
access_token_secret = 'cWwfsIC3dsixdbxV6HseTZUNkGAEABFHErFZZcz9sh7dZ'
password = "@ndych3n1454L46i5Z9"

# consumer_key = os.environ['CONSUMER_KEY']
# consumer_secret = os.environ['CONSUMER_SECRET']
# access_token = os.environ['ACCESS_TOKEN']
# access_token_secret = os.environ['ACCESS_TOKEN_SECRET']
# password = os.environ['PASSWORD']

def connect(created_at, tweet, retweet_count):
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
			query = "INSERT INTO TwitterSent (created_at, tweet, retweet_count) VALUES (%s, %s, %s)"
			cursor.execute(query, (created_at, tweet, retweet_count))
			con.commit()
			
			
	except Error as e:
		print(e)

	cursor.close()
	con.close()

	return


# Tweepy class to access Twitter API
class Streamlistener(tweepy.StreamListener):
	

	def on_connect(self):
		print("You are connected to the Twitter API")


	def on_error(self):
		if status_code != 200:
			print("error found")
			# returning false disconnects the stream
			return False

	"""
	This method reads in tweet data as Json
	and extracts the data we want.
	"""
	def on_data(self,data):
		
		try:
			raw_data = json.loads(data)

			if 'text' in raw_data:
				created_at = parser.parse(raw_data['created_at'])
				tweet = raw_data['text']
				retweet_count = raw_data['retweet_count']
				#insert data just collected into MySQL database
				connect(created_at, tweet, retweet_count)
				print("Tweet colleted at: {} ".format(str(created_at)))
		except Error as e:
			print(e)


if __name__== '__main__':

	# # #Allow user input
	# track = []
	# while True:

	# 	input1  = input("what do you want to collect tweets on?: ")
	# 	track.append(input1)

	# 	input2 = input("Do you wish to enter another word? y/n ")
	# 	if input2 == 'n' or input2 == 'N':
	# 		break
	
	# print("You want to search for {}".format(track))
	# print("Initialising Connection to Twitter API....")
	# time.sleep(2)

	# authentification so we can access twitter
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)
	api =tweepy.API(auth, wait_on_rate_limit=True)

	# create instance of Streamlistener
	listener = Streamlistener(api = api)
	stream = tweepy.Stream(auth, listener = listener)

	track = ['Tsla']
	#track = ['nba', 'cavs', 'celtics', 'basketball']
	# choose what we want to filter by
	stream.filter(track = track, languages = ['en'])