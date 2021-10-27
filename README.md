# SSIMSQL
Sentiment Scraper into a mySQL database (SSIMSQL)

This project will pull live data from reddit, finviz, interactive brokers, 
twitter, and the sec website into a mySQL database. The data consists of
sentiment from reddit, finviz, and twitter. Alongside some technical data 
from interactive brokers and finviz such as price, options data, and insider
trading. All of this data will be quantified numerically into one minute 
seqences. All of this will be strung together with multiprocessing as
it is needed to incorperate multiple data streams from different sources.

- SuperSent.py
  - strings all of the files together with multiprocessing

    - will truncate all of todays data and rest the dbs except for super db, where that is saved and serialized into hard drive. Then clean out super db.


- TSSmySQL.py
  - handles the twitter stream and stores it into a mySQL db

- RSSmySQL.py (1 param)
  - handles the data from TSSmySQL.py and condenses sentiment into
    one minute chunks

  - gets reddit sentiment (needs further research)
  - gets latest market news from finviz (needs  further research)
  - gets overall market sentiment from s&p 500 (Where would I get this data from?)
  - stores all of the above into a mysql db 
  - combines tweetdb with ibkrdb to form super db
            
- Ibkr.py (19 params)
    - gets price of stock
    - gets live stream of stock
    - technical analysis (VWAP, Bollinger Bands, RSI, *ema crossover??*)
    - support / resistance 1 min intervals
    - long term support / resistance with 1 day intervals
    - options data (gamma and theta?)  (How would I implement this?) 
    - IBKR live stock news. (Is it worth the money $250?)

      - place buy/sell orders here from 

- algo.py
    - will run a continual learning neural net that take the super db from RSS and give a buy/sell signal.

End Goal df

[timestamp, open, high, low, close, volume, bar count, bar size, bbHigh, bbLow, bbAvg, VWAP, RSI, RSIup, RSIdown, stSup, stRes, ltSup, ltRes, tweetSent, $GLD, $UVXY, $SQQQ]

current implemented size: 23

can add: ema cross over +2
         news sentiment from IBKR +1
         Options gamma, theta +2



find out what is an efficent way to store and wipe out all of the db after each day