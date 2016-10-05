#!/usr/bin/python
#import pandas as pd
import numpy as np
def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    
    cleaned_data = []

    ### your code goes here
    predictions2 = predictions.reshape([len(predictions),])
    net_worths2 = net_worths.reshape([len(net_worths),])
    errors = abs(predictions2 - net_worths2)
    indx = errors < np.percentile(errors, 90)
    for j, ind in enumerate(indx):
        if ind:
            cleaned_data.append((ages[j], net_worths[j], np.array([errors[j]])))
    return cleaned_data

