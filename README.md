# SSIMSQL
Sentiment Scraper into a mySQL database (SSIMSQL)

This project will pull data from reddit, finviz, interactive brokers, 
twitter, and the sec website into a mySQL database. The data consists of
sentiment from reddit, finviz, and twitter. Alongside some technical data 
from interactive brokers and finviz such as price, options data, and insider
trading. All of this data will be quantified numerically into one minute 
seqences. All of this will be strung together with multiprocessing as
it is needed to incorperate mmultiple data streams from different sources.


