#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split


clf = DecisionTreeClassifier()
clf.fit(features, labels)
print clf.score(features, labels)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.8, random_state=42)


clf_val = DecisionTreeClassifier()
clf_val.fit(X_train, y_train)
print clf_val.score(X_test, y_test)
### it's all yours from here forward!  

from sklearn.cross_validation import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(labels, 3, test_size=0.3, random_state=36)

features = np.array(features)
labels = np.array(labels)
for train_index, test_index in sss:
    train_index = train_index
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    clf_val = DecisionTreeClassifier()
    clf_val.fit(X_train, y_train)
    print "stratified split test:", clf_val.score(X_test, y_test)
