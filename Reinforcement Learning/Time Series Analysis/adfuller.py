

import pandas as pd
data=pd.read_csv(r'F:/Adv Python by sai/Datasets/daily-total-female-birth-in-cal.csv',header=0,index_col=0)

#convert dataframe into dataseries
x=data.values

split=round(len(x)/2)
print(len(data))
x1,x2=x[0:split],x[split:]

mean1,mean2=x1.mean(),x2.mean()

var1,var2=x1.var(),x2.var()



#to check data is stationary if mean and variance are almost same then stationary


print("mean1: ",mean1 ,"mean2: ",mean2)
print("variance1: ",var1 ,"variance2: ",var2)



# ADFuller Test (augmented dickey fuller test)

#null hypothesis(h0): time series is not stationary
# h1: time series is stationary

#if p-value <= 0.05 then  reject null hypothesis
#p-value > 0.05 then accept the null hypothesis

#use ADFuller to calculate statistics , p-value and critical value


from statsmodels.tsa.stattools import adfuller


result=adfuller(x)

print("ADF statistics %f"%result[0])

print("p value %f"%result[1])

print("critical values: ")

for k,v in result[4].items():
    print("\t%s : %0.3f"%(k,v))


if result[1]>0.05:
    print("accept null hypothesis,i.e time series is not stationary")
else:
    print("reject null hypothesis,i.e time series is  stationary")






