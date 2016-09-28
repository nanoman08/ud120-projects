#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#########################################################

from sklearn.svm import SVC

t0 = time()
linear_clf = SVC(kernel='linear')

linear_clf.fit(features_train, labels_train)

print "linear svc training time:", round(time()-t0, 3), "s"

t1 = time()
linear_clf.score(features_test, labels_test)

print "linear svc predict time:", round(time()-t1, 3), "s"

########################################################

features_train_100 = features_train[:len(features_train)/100] 
labels_train_100 = labels_train[:len(labels_train)/100] 

t0 = time()
linear_clf = SVC(kernel='linear')

linear_clf.fit(features_train_100, labels_train_100)

print "linear svc training time:", round(time()-t0, 3), "s"

t1 = time()
linear_clf.score(features_test, labels_test)

print "linear svc predict time:", round(time()-t1, 3), "s"


t0 = time()
gn_clf = SVC(kernel='rbf')

gn_clf.fit(features_train_100, labels_train_100)

print "gaussian svc training time:", round(time()-t0, 3), "s"

t1 = time()
gn_clf.score(features_test, labels_test)

print "guassian svc predict time:", round(time()-t1, 3), "s"


#########################################


t0 = time()
gn_clf = SVC(C = 1, kernel='rbf')

gn_clf.fit(features_train_100, labels_train_100)

print "gaussian svc training time:", round(time()-t0, 3), "s"

t1 = time()
print gn_clf.score(features_test, labels_test)

print "guassian svc predict time:", round(time()-t1, 3), "s"


gn_clf = SVC(C = 10, kernel='rbf')

gn_clf.fit(features_train_100, labels_train_100)

print gn_clf.score(features_test, labels_test)

gn_clf = SVC(C = 100, kernel='rbf')

gn_clf.fit(features_train_100, labels_train_100)

print gn_clf.score(features_test, labels_test)

gn_clf = SVC(C = 1000, kernel='rbf')

gn_clf.fit(features_train_100, labels_train_100)

print gn_clf.score(features_test, labels_test)

gn_clf = SVC(C = 10000, kernel='rbf')

gn_clf.fit(features_train_100, labels_train_100)

print gn_clf.score(features_test, labels_test)

gn_clf = SVC(C = 100000, kernel='rbf')

gn_clf.fit(features_train_100, labels_train_100)

print gn_clf.score(features_test, labels_test)

gn_clf_full = SVC(C = 10000, kernel='rbf')

gn_clf_full.fit(features_train, labels_train)

print gn_clf.score(features_test, labels_test)





