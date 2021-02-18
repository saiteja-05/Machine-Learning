

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import StackingClassifier

#it will generate data for models
def get_dataset():
    X,Y=make_classification(n_samples=1000,n_features=20,n_informative=15,
                            n_redundant=5,random_state=1)
    return X,Y


#it will generate the stacking model
def get_stacking():
    #base models
    level0=list()
    level0.append(('lr',LogisticRegression()))
    level0.append(('knn',KNeighborsClassifier()))
    level0.append(('dt',DecisionTreeClassifier()))
    level0.append(('svm',SVC()))
    level0.append(('gnb',GaussianNB()))
    
    level1=LogisticRegression() #meta model
    model=StackingClassifier( estimators=level0,final_estimator=level1,cv=5)
    return model

#get list of models to evaluate

def get_models():
    models={}
    models['lr']=LogisticRegression()
    models['knn']=KNeighborsClassifier()
    models['dt']=DecisionTreeClassifier()
    models['svm']=SVC()
    models['gnb']=GaussianNB()
    models['stacking']=get_stacking()
    return models


def evaluate_model(model,X,Y):
    cv=RepeatedStratifiedKFold(n_splits=10,n_repeats=3,random_state=1)
    scores=cross_val_score(model,X,Y,scoring='accuracy',cv=cv,n_jobs=-1,error_score='raise')
    return scores



X,Y=get_dataset()
model=get_models()

results,names=[],[]

for name,model in model.items():
    scores=evaluate_model(model,X,Y)
    results.append(scores)
    names.append(name)
    
#comparision of performance of different models
plt.boxplot(results,labels=names,showmeans=True)
plt.show()























































