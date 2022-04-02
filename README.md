# spyHedger
This application trades/predicts the s&p500 

- ___SuperSent.py___ 
  - Main app that uses multiprocessing to interconnect all of the files
    - ___SSmySQL.py___ 
      - gets live twitter data and stores into mysql
    - ___RSSmySQL.py___ 
      - gets reddit sentiment and stores into my sql
        - ___ibpy.py___
          - gets interactive brokers data (S&P data + more) and merges in SQL database
            - ___model.py___ 
              - uses a transformer neural network to make predicitons/trade (2 different models)
              - deploys trades based on preditiction/trade in live time

