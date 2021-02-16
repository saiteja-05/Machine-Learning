
print("Hello World!");




# import modules
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt 


# read data_set
data = pd.read_csv("Wage.csv")

data.head()
data_X=data[['age']]
data_Y=data['wage']


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(data_X,data_Y,test_size=0.30,random_state=1)

plt.scatter(X_train,Y_train,edgecolor='k',alpha=0.3)
plt.show()


from sklearn.linear_model import LinearRegression
model=LinearRegression()

model.fit(X_train,Y_train)
print(model.coef_)
print(model.intercept_)

y_pred=model.predict(X_test)
xp=np.linspace(X_test.min(),X_test.max(),70)
xp=xp.reshape(-1,1)
y_pred_plot=model.predict(xp)
plt.scatter(X_test,Y_test,edgecolor='k',alpha=0.3)
plt.plot(xp,y_pred_plot)
plt.show()




df_cut,bins=pd.cut(X_train['age'],4,retbins=True,right=True)
df_cut.value_counts(sort=False)


df_steps=pd.concat([X_train,df_cut,Y_train],keys=['age','age_cuts','wage'],axis=1)




# Create dummy variables for the age groups
df_steps_dummies = pd.get_dummies(df_cut)
df_steps_dummies.head()


df_steps_dummies.columns = ['17.938-33.5','33.5-49','49-64.5','64.5-80'] 

# Fitting Generalised linear models
fit3 = sm.GLM(df_steps.wage, df_steps_dummies).fit()

bin_mapping=np.digitize(X_test['age'],bins)

X_valid=pd.get_dummies(bin_mapping)

xp=np.linspace(X_test['age'].min(),X_test['age'].max(),70)

bin_mapping=np.digitize(xp,bins)

X_valid_2=pd.get_dummies(bin_mapping)

pred2 = fit3.predict(X_valid_2)
# Visualisation
fig, (ax1) = plt.subplots(1,1, figsize=(12,5))
fig.suptitle('Piecewise Constant', fontsize=14)

# Scatter plot with polynomial regression line
ax1.scatter(X_train, Y_train, facecolor='None', edgecolor='k', alpha=0.3)
ax1.plot(xp, pred2, c='b')

ax1.set_xlabel('age')
ax1.set_ylabel('wage')
plt.show()






