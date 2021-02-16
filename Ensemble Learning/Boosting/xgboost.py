

from sklearn.datasets import load_boston

boston=load_boston()
#to see the keys
print(boston.keys())

#to see number of columns and shape
print(boston.data.shape)

#to view the column names
print(boston.feature_names)

#to see description of dataset
print(boston.DESCR)


#convert data to DataFrame
import pandas as pd
import numpy as np
data=pd.DataFrame(boston.data)

data.columns=boston.feature_names

data.head()

data['Prices']=boston.target


#to understand data
data.info()
data.describe()


#pip install xgboost

import xgboost as xgb

from sklearn.metrics import mean_squared_error

x=data.iloc[:,:-1] #except last column
y=data.iloc[:,-1]  #only last column


#DMATRIX IS REPRESENTATION OF DATA INTERNALLY USED BY XGBOOST IT IMPROVES 
#IT IS MEMORY EFFICIENT WAY OF STORING THE DATA 

data_dmatrix=xgb.DMatrix(data=x,label=y)



from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.20,random_state=123)


xg_reg=xgb.XGBRegressor(objective='reg:linear',colsample_bytree=0.30,
                    learning_rate=0.1,max_depth=5,alpha=10,n_estimators=10)


xg_reg.fit(X_train,Y_train)

y_pred=xg_reg.predict(X_test)


rmse=(np.sqrt(mean_squared_error(Y_test,y_pred)))

print('RMSE %f'%(rmse))


params={
        "objective":"reg:linear",
        "colsample_bytree":0.30,
        "learning_rate":0.1,
        "max_depth":5,
        "alpha":10       
        }


cv_results=xgb.cv(dtrain=data_dmatrix,params=params,nfold=3,num_boost_round=50,
       early_stopping_rounds=10,metrics='rmse',as_pandas=True,seed=123)


cv_results.head()      

print((cv_results['test-rmse-mean']))
#print((cv_results['test-rmse-mean']).tail())



xg_reg=xgb.train(params=params,dtrain=data_dmatrix,num_boost_round=10)

import matplotlib.pyplot as plt

#conda install python-graphviz
xgb.plot_tree(xg_reg,num_trees=0)
plt.rcParams['figure.figsize']=[50,10]

plt.show()


xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize']=[5,5]

plt.show()








































