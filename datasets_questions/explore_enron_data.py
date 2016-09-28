#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
i= 0
j=0
k=0
for keys, items in enron_data.iteritems():
    if items['poi']:
        i+=1
    if items['email_address'] != 'NaN':
        j = j+1
    if items['salary'] != 'NaN':
        k = k+1

print i
    
for name in enron_data.keys():
    # query the dataset 1 and datast 2
    if 'prentice' in name.lower():
        print " {} total stock value owned by {}".format(enron_data[name]['total_stock_value'], name)
    
    if 'colwell' in name.lower():
        print " {} email from {} to POI:".format(enron_data[name]['from_this_person_to_poi'], name)
    
    if 'skilling' in name.lower():
        print " {} exercised stock options by {}:".format(enron_data[name]['exercised_stock_options'], name)
        print " {} toty payment by {}:".format(enron_data[name]['total_payments'], name)

    if 'fastow' in name.lower():
        print " {} exercised stock options by {}:".format(enron_data[name]['exercised_stock_options'], name)
        print " {} toty payment by {}:".format(enron_data[name]['total_payments'], name)

    if 'lay' in name.lower():
        print " {} exercised stock options by {}:".format(enron_data[name]['exercised_stock_options'], name)
        print " {} toty payment by {}:".format(enron_data[name]['total_payments'], name)


