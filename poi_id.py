#!/usr/bin/python

import sys
import pickle
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 0: Define functions I'll use since everything is in one script 

def create_features(data_dict):
    for key, value in data_dict.items():
        from_person_to_poi = value['from_this_person_to_poi']
        from_messages = value['from_messages']
        from_poi_to_this_person = value['from_poi_to_this_person']
        to_messages = value['to_messages']
        salary = value['salary']
        bonus = value['bonus']
        exercised_stock_options = value['exercised_stock_options']
        total_stock_value = value['total_stock_value']
        
        if from_person_to_poi != 'NaN' and from_messages != 'NaN':
            value['fract_to_poi'] = float(from_person_to_poi)/from_messages
        else:
            value['fract_to_poi'] = 'NaN'
        if from_poi_to_this_person != 'NaN' and to_messages != 'NaN':
            value['fract_from_poi'] = float(from_poi_to_this_person)/to_messages
        else:
            value['fract_from_poi'] = 'NaN'
        if salary != 'NaN' and bonus != 'NaN':
            value['salary_over_bonus'] = float(salary)/bonus
        else:
            value['salary_over_bonus'] = 'NaN'
        if exercised_stock_options != 'NaN' and total_stock_value != 'NaN' and total_stock_value != 0:
            value['exer_stock_opts_over_tot'] = float(exercised_stock_options)/total_stock_value
        else:
            value['exer_stock_opts_over_tot'] = 'NaN'
    return data_dict

def scale_features(data_dict):
    import pandas as pd
    
    
    # Create dataframe from dictionary
    data_df = pd.DataFrame.from_dict(data_dict, orient='index')
    data_df.replace('NaN', np.nan, inplace = True)
    
    # Scale using manual scaler to ignore non-numerical values
    scaled_df = data_df.apply(manual_MinMaxScaler)
    
    # Return NaN values to original form and create .csv to check
    scaled_df.replace(np.nan, 'NaN', inplace = True)
    scaled_df.to_csv('scaled_enron_data.csv')
    return pd.DataFrame.to_dict(scaled_df, orient='index')
    
def manual_MinMaxScaler(df):
    if df.name != 'name' and df.name != 'email_address' and df.name != 'poi':
        min_val = df.min()
        max_val = df.max()
        return (df-min_val)/(max_val-min_val)
    else:
        return df

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi', 'fract_from_poi', 'fract_to_poi',  
                'director_fees', 'restricted_stock_deferred', 
                'exercised_stock_options', 'expenses']
'''
features_list = ['poi', 'bonus', 'deferral_payments', 'deferred_income', 
                'director_fees', 'exercised_stock_options', 'expenses', 
                'from_messages', 'from_poi_to_this_person', 
                'from_this_person_to_poi', 'long_term_incentive', 'other',
                'restricted_stock', 'restricted_stock_deferred', 
                'salary_over_bonus', 'shared_receipt_with_poi', 'to_messages', 
                'total_payments', 'total_stock_value']
'''

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')
data_dict.pop('LOCKHART EUGENE E')

### Task 3: Create new feature(s)
new_data_dict = create_features(data_dict)

#### Scale features for use in PCA
new_data_dict = scale_features(new_data_dict)

### Store to my_dataset for easy export below.
my_dataset = new_data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from time import time

clf_DT = DecisionTreeClassifier(random_state=42)
pca = PCA(random_state = 42)

steps = [('PCA', pca),
         ('clf', clf_DT)]

pipe = Pipeline(steps)

cv = StratifiedShuffleSplit(100, random_state = 42)
params = {
        "PCA__n_components": range(2, 7)
}

gs = GridSearchCV(pipe, params, cv = cv, scoring = 'f1_weighted')


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


print "Training the classifier"
t0 = time()
gs.fit(features_train, labels_train)
#gs.fit(features, labels)
print "Training time: %0.3fs" % (time() - t0)

clf = gs.best_estimator_
pred = clf.predict(features_test)
print clf
print "Explained variance ratio:", gs.best_estimator_.named_steps['PCA'].explained_variance_ratio_
print "Feature importances:", gs.best_estimator_.named_steps['clf'].feature_importances_

print classification_report(labels_test, pred)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)