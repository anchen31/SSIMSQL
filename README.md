# spyHedger
This application trades/predicts the s&p500 

- ***SuperSent.py***
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

# spyHedger
This application trades/predicts the s&p500 

- ***SuperSent.py*** 
  - Main app that uses multiprocessing to interconnect all of the files
    - ***TSSmySQL.py*** 
      - gets live twitter data and stores into mysql
    - ***RSSmySQL.py*** 
      - gets reddit sentiment and stores into my sql
        - ***ibpy.py*** 
          - gets interactive brokers data (S&P data + more) and merges in SQL database
            - ***model.py*** 
              - uses a transformer neural network to make predicitons/trade (2 different models)
              - deploys trades based on preditiction/trade in live time




find out what is an efficent way to store and wipe out all of the db after each day