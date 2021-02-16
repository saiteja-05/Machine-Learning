
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.pylab import rcParams
#rcparams used for maintaing same configuration of figure by default which we define earlier
rcParams['figure.figsize']=15,6
from pandas import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
#dateparse=lambda dates: pd.datetime.strptime(dates,'%Y-%m')

from statsmodels.tsa.api import ExponentialSmoothing,SimpleExpSmoothing,Holt

#time=pd.read_excel('AirpassengersData.xls',parse_dates=['Month'],index_col='Month',date_parser=dateparse)

#time.head()

df=pd.read_csv('F:/Adv Python by sai/Datasets/Airpassengers.csv',parse_dates=['Month'],index_col=['Month'])

df.head()

time1=df['#Passengers']
time1.head()


time1.plot(kind="line",figsize=(10,5))



time_log = np.log(time1)
time_log.plot(kind="line",figsize=(10,5))


decomposition = seasonal_decompose(time_log)

decomposition.plot()



ses = SimpleExpSmoothing(time_log).fit(smoothing_level=0.6,optimized=False)
ses1 = ses.forecast(len(time_log))


time_log.plot(kind="line",figsize=(10,5))
ses1.plot(kind="line",figsize=(10,5),color='orange')


#Holt's winter method or triple decomposition
#as we consider trend and seasonality for prediction
#if we consier only trend then it is called as Hol't method
ets_stl = ExponentialSmoothing((time_log) ,seasonal_periods=12 ,trend='add', seasonal='add').fit()
ets_stl1 = ets_stl.forecast(len(time_log))



time_log.plot(kind="line",figsize=(10,5),legend=True)
ets_stl1.plot(kind="line",figsize=(10,5),color='orange',legend=True,label='ETS pred')