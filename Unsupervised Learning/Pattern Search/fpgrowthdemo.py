

import numpy as np
import pandas as pd
import pyfpgrowth as fp


#transactions=[[1,2,5],[2,4],[2,3],[1,2,4],[1,3],[2,3],[1,3],
#              [1,2,3,5],[1,2,3]]


dataset=pd.read_csv('Datasets/Market.csv',header=None)
transactions =[]

for sublist in dataset.values.tolist():
    clean_sublist=[item for item in sublist if item is not np.nan]
    transactions.append(clean_sublist)
    




#2 is support threshold
#0.7 is confidence threshold


patterns=fp.find_frequent_patterns(transactions,2)
rules=fp.generate_association_rules(patterns,0.7)



