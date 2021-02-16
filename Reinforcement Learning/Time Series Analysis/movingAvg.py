


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')


datafile = 'F:\Adv Python by sai\Datasets\movavg.csv'
datam=pd.read_csv(datafile,index_col= 'Date')
datam.index = pd.to_datetime(datam.index)


weights=np.arange(1,11)
weights


datam.columns

sma10 = datam['Price'].rolling(10).mean()
sma10
wma10=datam['Price'].rolling(10).apply(lambda prices:np.dot(prices, weights)/weights.sum(),raw=True)

wma10.head(20)

#datam['10 day WMA'] = np.round(wma10, decimals=3)
datam['Our 10 day WMA'] = np.round(wma10, decimals=3)

datam[['Price', '10-day WMA','Our 10 day WMA']].head(20)



from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

sma10 = datam['Price'].rolling(10).mean()
plt.figure(figsize = (12,6))
plt.plot(datam['Price'],label="Price")
plt.plot(wma10,label='10-Day WMA')
plt.plot(sma10,label='10-Day Sma')
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()


ema10=datam['Price'].ewm(span=10,adjust=False).mean()


ema10.head()

datam['Our 2nd 10-day EMA'] = np.round(ema10,decimals=3)

datam[['Price','10-day EMA','Our 10-day EMA','Our 2nd 10-day EMA']].head(20)

#ema10alt=modPrice.ewm(span=10,adjust=False).mean()
#ema10alt

sma10 = datam['Price'].rolling(10).mean()
plt.figure(figsize = (12,6))
plt.plot(datam['Price'],label="Price")
plt.plot(wma10,label='10-Day WMA')
plt.plot(sma10,label='10-Day Sma')
plt.plot(ema10,label='10-day ema-1')
plt.plot(ema10,label='10-day ema-2')
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()




