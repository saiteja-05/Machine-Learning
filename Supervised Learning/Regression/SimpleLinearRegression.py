

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#simple linear regression

#step1:gather data
dataset=pd.read_csv("Datasets\Salary_Data.csv")
X=dataset.iloc[:,:-1]
Y=dataset.iloc[:,1]


#step2:split data into train and test 
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)


# =============================================================================
# #step3:Feature Scaling (why ?) not in proper scale proportion so we nedd to do
# #normalize data then we use feature scaling
# from sklearn.preprocessing import StandardScaler
# sc_X=StandardScaler()
# X_train=sc_X.fit_transform(X_train)
# X_test=sc_X.transform(X_test)
# 
# sc_Y=StandardScaler()
# Y_train=sc_Y.fit_transform(Y_train)
# Y_test=sc_Y.transform(Y_test)
# 
# =============================================================================

#step3: import linear model and linear regression class


from sklearn.linear_model import LinearRegression

regressor=LinearRegression();
regressor.fit(X_train,Y_train)  #internally it uses gradient descendent

regressor.coef_


Y_pred=regressor.predict(X_test) #predicted values

#for manual values
i=np.array([1,2,3,4]).reshape(-1,1) #2-d array of size 1 row
i_out=regressor.predict(i)

#for testing data
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.title("Experience vs Salary Testing Data")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()


#for training data
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Experience vs Salary Training Data")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()



from sklearn import metrics
print("mean squared error",metrics.mean_squared_error(Y_test,Y_pred))
print("mean Absolute error",metrics.mean_absolute_error(Y_test,Y_pred))


























