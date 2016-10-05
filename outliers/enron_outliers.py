#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot as plt
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop('TOTAL',0)
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

### your code below
salary, bonus = zip(*list(data))
plt.scatter(salary,bonus, color = 'Blue')
plt.xlabel('salary')
plt.ylabel('bonus')
plt.show()

