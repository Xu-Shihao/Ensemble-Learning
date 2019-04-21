# Ensemble Learning Template

Ensemble Learning (concrete feature selection, Parameter Tuning, and Oversampling) with Cross-validation

Multiple classifiers are implemented in the code:
1. logistic regression
2. SVM
3. Gradient boost
4. Adaboost
5. RandomForest

## Diagram

[[https://ibb.co/BsRJMVy]]

## Getting Started

Ensemble learning: Soft Voting by Multiple Classifiers

In every cross-validation loop, we seperat samples as training data and testing data. 10-fold cross-validation was formed on training data to determine the best parameters and number of top features for each classifier. Finally, training data were fitted by multiple optimized classifiers, and then soft voted the prediction scores of testing data with different weights (late fusion).

Late fusion: treat features in each feature set individually, and soft vote the final prediction results of each classifier and each feature set. This approach can be useful if the number of features in each dataset varies widely.

All metrics in each cross-validation are saved in ./tmp folder. Final results can be reproduced based on test_combine.py

### Prerequisites

Please notice that: this code is only working on a binary classification task, and the feature should be numerical.

What things you need:

```
Python 3.x
scikit-learn == 0.19.2
imbalance-learn == 0.3.3
numpy==1.14.5
```

Please also follow the Example.csv file in ./features folder to form your own data

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

## Running the code

```
python main_classification.py
python test_combine.py
```

## Addition

1. you can change the number of cross-validation fold in line 148-152 of main_classification.py

```
    # skf=StratifiedKFold(n_splits=10)
    # for train_index, test_index in skf.split(range(len(LoadFeatures_pool[0])),doc_label):
    loo=LeaveOneOut()
    for train_index, test_index in loo.split(range(len(LoadFeatures_pool[0]))):
```

2. By default in this code, above-mentioned 5 different classifiers are used to predict the test data, and without feature selection (percentile=100).

You can enable feature selection and parameter tuning by uncommenting line 301-302 and commenting line 304-312 in main_classification.py.

The CV_Para_selection() function contains feature selection and parameter tuning:

a. feature selection

```
para_steps.update({pipeline_step1: [SelectPercentile(feature_selection.f_classif)],
                               pipeline_step1 + '__percentile': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                               }
                              )
```
b. parameter tuning
```
      if cf_item == 'Logistic Regression':
            para_steps.update({pipeline_step2: [get_model('Logistic Regression')],
                               pipeline_step2 + '__C': [0.01, 0.1, 1],
                               pipeline_step2 + '__solver': [ 'liblinear', 'sag']
                               }
                              )
        elif cf_item == 'SVM':
            para_steps.update({pipeline_step2: [get_model('SVM')],
                               pipeline_step2 + '__kernel': ['linear', 'rbf'],
                               pipeline_step2 + '__C': [0.1, 0.5, 1],
                               pipeline_step2 + '__gamma': [0.1, 0.01, 0.001]
                               }
                              )
        elif cf_item == 'Gradient Boosting':
            para_steps.update({pipeline_step2: [get_model('Gradient Boosting')],
                               pipeline_step2 + '__learning_rate': [0.1, 0.5, 1],
                               pipeline_step2 + '__max_depth': [3, 4, 5, 7],
                               pipeline_step2 + '__n_estimators': [100, 150, 200]
                               }
                              )
        elif cf_item == 'AdaBoost':
            para_steps.update({pipeline_step2: [get_model('AdaBoost')],
                               pipeline_step2 + '__learning_rate': [0.1, 0.5, 1],
                               pipeline_step2 + '__n_estimators': [50, 100, 150]
                               }
                              )
        elif cf_item == 'RandomForest':
            para_steps.update({pipeline_step2: [get_model('RandomForest')],
                               pipeline_step2 + '__n_estimators': [100, 150, 200],
                               pipeline_step2 + '__max_depth': [3, 5, 7, None]
                               }
                              )
        elif cf_item == 'MLP':
            para_steps.update({pipeline_step2: [get_model('MLP')],
                               pipeline_step2 + '__hidden_layer_sizes': [100, 150, 200],
                               pipeline_step2 + '__activation': ['identity', 'logistic', 'tanh', 'relu']
                               }
                              )
```

3. if you enable CV_Para_selection() function, in line 151 in sklearn_classifier.py, you need to take care two parameters of GridSearchCV() function:

cv: how many fold applied to choose the best parameter

n_jobs: how many CPUs are used to parallelly run the code, higher -> faster


## reference
If you use this code for research purpose, please cite one of the following articles

[1]. S. Xu, Z. Yang, D. Chakraborty, Y. Tahir, T. Maszczyk, Y. H. V. Chua, J. Dauwels, D. Thalmann, N. M. Thalmann, B. L. Tan, J. C. K. Lee, "Automated Verbal and Non-verbal Speech Analysis of Interviews of Individuals with Schizophrenia and Depression", 41th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC'19)

## Authors

XU SHIHAO
Nanyang Technological Univesity, Singapore

