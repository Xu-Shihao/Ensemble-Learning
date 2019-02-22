# Ensemble Learning Template

Ensemble Learning (concrete feature selection, Parameter Tuning, and Oversampling) with Cross-validation

Multiple classifiers are implemented in the code:
1. logistic regression
2. SVM
3. Gradient boost
4. Adaboost
5. RandomForest

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

1. Ensemble learning: Soft Voting by Multiple Classifiers

In every cross-validation loop, we seperat samples as training data and testing data. 10-fold cross-validation was formed on training data to determine the best parameters and number of top features for each classifier. Finally, training data were fitted by multiple optimized classifiers, and then soft voted the prediction scores of testing data with different weights (late fusion).

Note：
Late fusion: treat features in each feature set individually, only soft voting the final prediction results. This approach can be useful if the number of features in each dataset varies widely。



### Prerequisites

Please notice that: this code is only working on a binary classification task, and the feature should be numerical.


What things you need:

```
Python 3.x
scikit-learn == 0.19.2
imbalance-learn == 0.3.3
numpy==1.14.5
```

Please also follow the Example.csv file in ./data folder to form your own data

The data should be like:

```
name of file: FeatureSetName_xxx.csv
ID feature_name_1 feature_name_2 .... feature_name_n Label
1        x             x                   x           A
2        x             x                   x           B
3        x             x                   x           A
.
.
m        x             x                   x           B
```



### Installing

Just install the import packages using:  pip install package_name

Then change the input/output address in main_classification.py

## Running the code

```
python main_classification 
```

## Addition

1. you can change the number of cross-validation fold in line 148-152 of main_classification.py

'''
    # skf=StratifiedKFold(n_splits=10)
    # for train_index, test_index in skf.split(range(len(LoadFeatures_pool[0])),doc_label):
    loo=LeaveOneOut()
    for train_index, test_index in loo.split(range(len(LoadFeatures_pool[0]))):
'''

2. By default in this code, above-mentioned 5 different classifiers are used to predict the test data, and without feature selection (percentile=100).

You can enable feature selection and parameter tuning by uncommenting line 301-302 and commenting line 304-312 in main_classification.py.

The CV_Para_selection() function contains feature selection and parameter tuning:

a. feature selection

'''
para_steps.update({pipeline_step1: [SelectPercentile(feature_selection.f_classif)],
                               pipeline_step1 + '__percentile': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                               }
                              )
'''



## Authors

XU SHIHAO
Nanyang Technological Univesity, Singapore

