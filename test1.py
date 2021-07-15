import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

ticker = "tsla"

# Getting Finviz Data
news_tables = {}        # contains each ticker headlines
url = f'https://finviz.com/quote.ashx?t={ticker}'
req = Request(url=url, headers={'user-agent': 'news'})
response = urlopen(req)     # taking out html response
        
html = BeautifulSoup(response, features = 'html.parser')
news_table = html.find(id = 'news-table') # gets the html object of entire table
news_tables[ticker] = news_table

print(news_table[ticker])    
    
# Parsing and Manipulating
parsed = []    
for ticker, news_table in news_tables.items():  # iterating thru key and value
    for row in news_table.findAll('tr'):  # for each row that contains 'tr'
        title = row.a.text
        source = row.span.text
        date = row.td.text.split(' ')
        if len(date) > 1:     # both date and time, ex: Dec-27-20 10:00PM
            date1 = date[0]
            time = date[1]
        else:time = date[0] # only time is given ex: 05:00AM
        

# Applying Sentiment Analysis
df = pd.DataFrame(parsed, columns=['Ticker', 'date', 'Time', 'Title'])

# for every title in data set, give the compund score
score = lambda title: sia.polarity_scores(title)['compound']
df['compound'] = df['Title'].apply(score)   # adds compund score to data frame

# Visualization of Sentiment Analysis
df['date'] = pd.to_datetime(df.date).dt.date # takes date comlumn convert it to date/time format

print(df)
