#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 

from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split


clf = DecisionTreeClassifier()
clf.fit(features, labels)
print clf.score(features, labels)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

clf_val = DecisionTreeClassifier()
clf_val.fit(X_train, y_train)
print clf_val.score(X_test, y_test)
sum(clf_val.predict(X_test))  # total predict poi number

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score

print confusion_matrix(y_test, clf_val.predict(X_test))
print precision_score(y_test, clf_val.predict(X_test))
print recall_score(y_test, clf_val.predict(X_test))
