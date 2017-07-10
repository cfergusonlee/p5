#!/usr/bin/python

import sys
import pickle
import numpy as np
#sys.path.append("../tools/")

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

features_list = ['poi', 'salary', 'bonus', 'fract_to_poi', 'fract_from_poi',
                'director_fees', 'restricted_stock_deferred', 
                'exercised_stock_options', 'expenses', 'total_stock_value']

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
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, KFold
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
from time import time






### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


# Classifiers
class_weights = {
                0.0: 18.0/143, 
                1.0: 125.0/143}
clf_SVC = SVC(
            class_weight = 'balanced',
            random_state = 42)

# PCA
pca = PCA(random_state = 42)

# Pipeline
steps = [
        ('clf', clf_SVC)]
pipe = Pipeline(steps)

# GridSearchCV
cv = StratifiedShuffleSplit(100, random_state = 42)
params = {
        'clf__kernel': ['poly', 'rbf'],
        'clf__degree': [2, 3, 4]}
gs = GridSearchCV(
    pipe,
    params,
    scoring = 'f1_weighted')

'''
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
'''
np_features = np.array(features)
np_labels = np.array(labels)

t0 = time()
sss = StratifiedShuffleSplit(n_splits = 10, test_size = .5, random_state = 42)
sss.get_n_splits(np_features, np_labels)
accuracy_scores = []
print "Training the classifier"

for train_index, test_index in sss.split(np_features, np_labels):
    features_train, features_test = np_features[train_index], np_features[test_index]
    labels_train, labels_test = np_labels[train_index], np_labels[test_index]

    # Train Classifier
    gs.fit(features_train, labels_train)
    clf = gs.best_estimator_
    print "Training time: %0.3fs" % (time() - t0)
    # Extract best classifier
    pred = clf.predict(features_test)
    print clf
    print classification_report(labels_test, pred)
    accuracy_scores.append(accuracy_score(pred, labels_test))


print "Overall accuracy:", np.mean(accuracy_scores)

print "Confusion Matrix:"
print "POI vs Non-POI"
print confusion_matrix(labels_test, pred, labels=[1, 0])
#print "Explained variance ratio:", gs.best_estimator_.named_steps['PCA'].explained_variance_ratio_



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)