

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#data preprocessing
dataset=pd.read_csv("Datasets/Market.csv",header=None)
transactions=[]

for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
    

#training the apriori on the dataset

from apyori import apriori

rules=apriori(transactions,
              min_support=0.003,
              min_confidence=0.2,
              min_lift=3,
              min_length=2)


#visualizing the results

MB=list(rules)
Result=[list(MB[i][0]) for i in range(0,len(MB))]




