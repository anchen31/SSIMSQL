# spyHedger
This application trades/predicts the s&p500 

SuperSent.py - Main app that uses multiprocessing to interconnect all of the files
  |
  | - TSSmySQL.py - gets live twitter data and stores into mysql
  | - RSSmySQL.py - gets reddit sentiment and stores into my sql
        |
        | - ibpy.py - gets interactive brokers data (S&P data + more) and merges in SQL database
              |
              |- model.py - uses a transformer neural network to make predicitons/trade (2 different models)
                          - deploys trades based on preditiction/trade in live time





