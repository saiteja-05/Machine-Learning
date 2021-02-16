

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns

#with out dimensionaity reduction

breast_cancer=load_breast_cancer()
X=pd.DataFrame(breast_cancer.data,columns=breast_cancer.feature_names)
#Y=pd.Categorical.from_codes(breast_cancer.target,breast_cancer.target_names)
y=pd.DataFrame(breast_cancer.target)



#TRAIN TEST SPLIT
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.30,random_state=0)


#design the model
knn=KNeighborsClassifier(n_neighbors=5,metric="euclidean")
knn.fit(X_train,Y_train)
y_pred=knn.predict(X_test)


#df=X_test.join(Y_test)

#sns.scatterplot(x='mean area',y='mean compactness',data=X_test.join(Y_test))
#using scatter plot
plt.scatter(X_test['mean area'],X_test['mean compactness'],c=y_pred,cmap='coolwarm',alpha=0.7)

#for confusion_matrix
print(confusion_matrix(Y_test,y_pred)) 

#accuracy score
from sklearn.metrics import accuracy_score
print("training data:",accuracy_score(Y_train,knn.predict(X_train)))
print("testing data:",accuracy_score(Y_test,y_pred))

#----------------------------------------------------------------------------



#use pca reduction and generate knn model after that generate accuracy_score
from sklearn.decomposition import PCA
pca=PCA(n_components=1)
X_train=pca.fit_transform(X_train)
X_test=pca.fit_transform(X_test)

#design the model
#knn=KNeighborsClassifier(n_neighbors=5,metric="euclidean")
knn.fit(X_train,Y_train)
y_pred=knn.predict(X_test)

print(confusion_matrix(Y_test,y_pred)) #for confusion_matrix

#accuracy score
from sklearn.metrics import accuracy_score
print("training data:",accuracy_score(Y_train,knn.predict(X_train)))
print("testing data:",accuracy_score(Y_test,y_pred))


#----------------------------------------------------------------------------

#use lda and generate knn model and generate accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lad=LDA(n_components=1)
X_train=lad.fit_transform(X_train,Y_train) #changed to single dimension
X_test=lad.fit_transform(X_test,Y_test)


#design the model
knn=KNeighborsClassifier(n_neighbors=5,metric="euclidean")
knn.fit(X_train,Y_train)
y_pred=knn.predict(X_test)

print(confusion_matrix(Y_test,y_pred)) #for confusion_matrix

#accuracy score
from sklearn.metrics import accuracy_score
print("training data:",accuracy_score(Y_train,knn.predict(X_train)))
print("testing data:",accuracy_score(Y_test,y_pred))




#----------------------------------------------------------------------------

#use QDA and generate knn model and generate accuracy_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

model=QDA()
model.fit(X_train,Y_train)
model.predict(X_test)

#using scatter plot
plt.scatter(X_test['mean area'],X_test['mean compactness'],c=y_pred,cmap='coolwarm',alpha=0.7)

#for confusion_matrix
print(confusion_matrix(Y_test,y_pred)) 

#accuracy score
from sklearn.metrics import accuracy_score
print("training data:",accuracy_score(Y_train,model.predict(X_train)))
print("testing data:",accuracy_score(Y_test,y_pred))









#--------------------------------------------------------------------------























