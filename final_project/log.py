# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 17:11:21 2016

@author: CHOU_H
"""

def classifier_bin_plot(data, y_name, key, bin_step):
    """
    Print out selected statistics regarding y, given a feature of
    interest
    """
    
    # Check that the key exists
    if key not in data.columns.values :
        print "'{}' is not a feature of the Titanic data. Did you spell something wrong?".format(key)
        return False

    # Return the function before visualizing if 'Cabin' or 'Ticket'
    # is selected: too many unique categories to display



    all_data = data
    


    # Create outcomes DataFrame
    all_data = all_data[[key, y_name]]
    
    # Create plotting figure
    plt.figure(figsize=(8,6))

    # 'Numerical' features

        
        # Remove NaN values from Age data
    all_data = all_data[~np.isnan(all_data[key])]
        
        # Divide the range of data into bins and count survival rates
    min_value = all_data[key].min()
    max_value = all_data[key].max()
    value_range = max_value - min_value


    bins = np.arange(min_value, mex_value + 10, value_range/bin_step )
        
        # Overlay each bin's survival rates
    y0 = all_data[all_data[y_name] == 0][key].reset_index(drop = True)
    y1 = all_data[all_data[y_name] == 1][key].reset_index(drop = True)
    plt.hist(y0, bins = bins, alpha = 0.6, color = 'red', label = 'y0')
        plt.hist(y1, bins = bins, alpha = 0.6, color = 'green', label = 'y1')
    
        # Add legend to plot
    plt.xlim(0, bins.max())
    plt.legend(framealpha = 0.8)
    
    # 'Categorical' features
#    else:
#       
#        # Set the various categories
#        if(key == 'Pclass'):
#            values = np.arange(1,4)
#        if(key == 'Parch' or key == 'SibSp'):
#            values = np.arange(0,np.max(data[key]) + 1)
#        if(key == 'Embarked'):
#            values = ['C', 'Q', 'S']
#        if(key == 'Sex'):
#            values = ['male', 'female']
#
#        # Create DataFrame containing categories and count of each
#        frame = pd.DataFrame(index = np.arange(len(values)), columns=(key,'Survived','NSurvived'))
#        for i, value in enumerate(values):
#            frame.loc[i] = [value, \
#                   len(all_data[(all_data['Survived'] == 1) & (all_data[key] == value)]), \
#                   len(all_data[(all_data['Survived'] == 0) & (all_data[key] == value)])]
#
#        # Set the width of each bar
#        bar_width = 0.4
#
#        # Display each category's survival rates
#        for i in np.arange(len(frame)):
#            nonsurv_bar = plt.bar(i-bar_width, frame.loc[i]['NSurvived'], width = bar_width, color = 'r')
#            surv_bar = plt.bar(i, frame.loc[i]['Survived'], width = bar_width, color = 'g')
#
#            plt.xticks(np.arange(len(frame)), values)
#            plt.legend((nonsurv_bar[0], surv_bar[0]),('Did not survive', 'Survived'), framealpha = 0.8)

    # Common attributes for plot formatting
    plt.xlabel(key)
    plt.ylabel('Counts')
    plt.title('key Statistics With \'%s\' Feature'%(key))
    plt.show()

#    # Report number of passengers with missing values
#    if sum(pd.isnull(all_data[key])):
#        nan_outcomes = all_data[pd.isnull(all_data[key])]['Survived']
#        print "Passengers with missing '{}' values: {} ({} survived, {} did not survive)".format( \
#              key, len(nan_outcomes), sum(nan_outcomes == 1), sum(nan_outcomes == 0)