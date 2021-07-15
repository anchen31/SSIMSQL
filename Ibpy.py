from ib_insync import *

ib = IB()
ib.connect(host='127.0.0.1', port=7497, clientId=1)

contract = Stock('TSLA', 'SMART', 'USD')

data = ib.reqMktData(contract)
print(data.marketPrice())

bars = ib.reqHistoricalData(
	contract, 
	endDateTime='',
    durationStr='1 D',
    barSizeSetting='1 min',
    whatToShow='TRADES',
    useRTH=False,
    formatDate=1,
    keepUpToDate=True)

