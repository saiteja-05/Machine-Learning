
print("Hello World!");

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import seaborn as sns
train = pd.read_csv('titanic_train.csv')
train.isnull()
train.info()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')

def replace_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
train['Age'] = train[['Age','Pclass']].apply(replace_age,axis=1)

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

train.drop('Cabin',axis=1,inplace=True)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train)
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,
              palette='RdBu_r')
#find how many survived from each passenger class
sns.countplot(x='Survived',hue='Pclass',data=train,
              palette='RdBu_r')
sns.distplot(train['Age'],kde=False,color='darkred',bins=40)



#convert categorical columns into numeric values
#dummy value trap
sex=pd.get_dummies(train['Sex'],drop_first=True)
embark=pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],inplace=True,axis=1)
train=pd.concat([train,sex,embark],axis=1)
X=train.drop('Survived',axis=1)
Y=train['Survived']


#splitting the data 
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30,random_state=0)


from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,Y_train) #for training the model
predictions=logmodel.predict(X_test) #for testing the data

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(Y_test,predictions))
print(confusion_matrix(Y_test,predictions))








