

#design  the arima model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller 
from statsmodels.tsa.seasonal import seasonal_decompose
#from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_model import ARIMA

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

# def parser(x):
# 	return datetime.strptime('190'+x, '%Y-%m')
 
# series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)


df=pd.read_csv(r"F:\Adv Python by sai\Datasets\AirPassengers.csv",parse_dates=['Month'],index_col=['Month'])

df.head()

#plt the data
plt.xlabel("dates")
plt.ylabel("number of passengers")

plt.plot(df)


#moving average and variance checking for time series is stationary or not
#if you see the graph moving average and std is not almost horizontal lines so it is changing 
#over the period 
#so the time series is not stationary




rolling_mean=df.rolling(window=12).mean()
rolling_stddev=df.rolling(window=12).std()
plt.xlabel("dates")
plt.ylabel("number of passengers")
plt.plot(df,color='blue',label='data')
plt.plot(rolling_mean,color='red',label='Rolling Mean')
plt.plot(rolling_stddev,color='green',label="Rolling Std")
plt.legend(loc='best')
plt.show()


#adfuller usage
#to check whether time series is stationary you can use adfuller test for hypothesis testing
#since p value is > 0.05 so accept null hypothesis that time series is not stationary
X=df['#Passengers']
result=adfuller(X)

print("ADF statistics %f"%result[0])

print("p value %f"%result[1])

print("critical values: ")

for k,v in result[4].items():
    print("\t%s : %0.3f"%(k,v))


if result[1]>0.05:
    print("accept null hypothesis,i.e time series is not stationary")
else:
    print("reject null hypothesis,i.e time series is  stationary")






def get_stationary(tmseries):
    rolling_mean=tmseries.rolling(window=12).mean()
    rolling_stddev=tmseries.rolling(window=12).std()
    plt.xlabel("dates")
    plt.ylabel("number of passengers")
    plt.plot(tmseries,color='blue',label='data')
    plt.plot(rolling_mean,color='red',label='Rolling Mean')
    plt.plot(rolling_stddev,color='green',label="Rolling Std")
    plt.legend(loc='best')
    plt.show()


    #adfuller usage
    X=tmseries['#Passengers']

    result=adfuller(X)

    print("ADF statistics %f"%result[0])

    print("p value %f"%result[1])

    print("critical values: ")

    for k,v in result[4].items():
        print("\t%s : %0.3f"%(k,v))


    if result[1]>0.05:
        print("accept null hypothesis,i.e time series is not stationary")
    else:
        print("reject null hypothesis,i.e time series is  stationary")

    



# as the data is not stationary we need to normalize

#to convert frame to stationay data
df_log=np.log(df)
plt.plot(df_log)

rolling_mean=df_log.rolling(window=12).mean()
df_log_minus_mean=df_log-rolling_mean #gives error
df_log_minus_mean.dropna(inplace=True)

get_stationary(df_log_minus_mean) #applyig function to check stationary or not

#using ewm exponential weighted mean
rolling_mean_exp_decay=df_log.ewm(halflife=12,min_periods=0,adjust=True).mean()

df_rolling_exp_decay=df_log-rolling_mean_exp_decay

df_rolling_exp_decay.dropna(inplace=True)

get_stationary(df_log_minus_mean)



#null,(x1-x0),(x2-x1),(x3-x2)
#subtracting previous period data from current time data

df_log_shift=df_log-df_log.shift() #1-day mvg avg

#same thing can be done using np.diff
#y_diff=np.diff(df_log)

df_log_shift.dropna(inplace=True)
get_stationary(df_log_shift)


decompose=seasonal_decompose(df_log)
decompose.plot()


#Plotting ACF & PACF 

#ACF & PACF plots
from statsmodels.tsa.stattools import acf,pacf
lag_acf = acf(df_log_shift, nlags=20)
lag_pacf = pacf(df_log_shift, nlags=20, method='ols')

#Plot ACF:
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_log_shift)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(df_log_shift)), linestyle='--', color='gray')
plt.title('Autocorrelation Function')            

#Plot PACF
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_log_shift)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(df_log_shift)), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function')
            
plt.tight_layout()            


# From the ACF graph, we see that curve touches y=0.0 line at x=2. Thus, from 
#theory, Q = 2 
#From the PACF graph, we see that curve touches y=0.0 line at x=2. 
#Thus, from theory, P = 2

# ARIMA is AR +  I +MA. Before, we see an ARIMA model, let us check the results of the individual AR & MA model. Note that, these models will give a value of RSS. Lower RSS values indicate a better model.





#AR Model
#making order=(2,1,0) gives RSS=1.5023
model = ARIMA(df_log, order=(2,1,0))
results_AR = model.fit(disp=-5)
plt.plot(df_log_shift)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'%sum((results_AR.fittedvalues - df_log_shift['#Passengers'])
                                                                         **2))
print('Plotting AR model')



#MA Model
model = ARIMA(df_log, order=(0,1,2))
results_MA = model.fit(disp=-1)
plt.plot(df_log_shift)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'%sum((results_MA.fittedvalues - df_log_shift['#Passengers'])**2))
print('Plotting MA model')






# AR+I+MA = ARIMA model
model = ARIMA(df_log, order=(2,1,2))
results_ARIMA = model.fit(disp=-1)
plt.plot(df_log_shift)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'%sum((results_ARIMA.fittedvalues - df_log_shift['#Passengers'])**2))
print('Plotting ARIMA model')

# By combining AR & MA into ARIMA, we see that RSS value has decreased from either case to 1.0292, indicating ARIMA to be better than its individual component models.

# With the ARIMA model built, we will now generate predictions. But, before we do any plots for predictions ,we need to reconvert the predictions back to original form. This is because, our model was built on log transformed data.


#prediction and transform
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head())

#Convert to cumulative sum
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum)


#copy only first value in all rows and then add cumulative sum in the value

predictions_ARIMA_log = pd.Series(df_log['#Passengers'].iloc[0], 
                                                    index=df_log.index)

#this is need to undo the effect of intgration using order 1 

predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,
                                                  fill_value=0)
predictions_ARIMA_log.head()

#this is needed for undo the logarithmic scale
# Inverse of log is exp.
predictions_ARIMA = np.exp(predictions_ARIMA_log)
#plotting against actual and predicted values
plt.plot(df)
plt.plot(predictions_ARIMA)



#We have 144(existing data of 12 yrs in months) data points. 
#And we want to forecast for additional 120 data points or 10 yrs.
results_ARIMA.plot_predict(1,264) 
x=results_ARIMA.forecast(steps=120)


# invert the differenced forecast to something usable

#print(np.exp(x))
#print(x[1])
#print(len(x[1]))
#print(np.exp(x[1]))