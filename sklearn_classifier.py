import csv
import os
from os.path import basename
import statistics
from collections import Counter
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn import feature_selection
from sklearn import metrics
from sklearn import preprocessing
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# load data
def open_file(file_name):
    '''
    This function is aiming to open the .csv file as following structure, and return the
    features set array, label array and the list of feature names.

    ID         feature-1  feature-2  feature-3  feature-4  ... feature-n Label
    sample1      xxx         xxx        xxx        xxx           xxx       a
    sample2      xxx         xxx        xxx        xxx           xxx       a
    sample3      xxx         xxx        xxx        xxx           xxx       b
    ...          ...         ...        ...        ...           ...      ...
    samplen      xxx         xxx        xxx        xxx           xxx       b

    '''
    ini = 0
    ClassLabel = []
    Data = []
    with open(file_name, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        # first line is feature names, last column is label
        for row in spamreader:
            if ini == 0:
                FeatureNames = row[1:-1]
            else:
                ClassLabel.append(row[-1])
                Data.append([float(value) for value in row[1:-1]])
            ini = ini + 1

    # Identify non-numeric label
    if ~ClassLabel[0].isdigit():
        le = preprocessing.LabelEncoder()
        le.fit(ClassLabel)
        ClassLabel = le.transform(ClassLabel)

    # return numpy format
    X_tensor = np.array(Data)
    Y_tensor = np.array(ClassLabel)
    return X_tensor, Y_tensor, FeatureNames

def CV_Para_selection(X_tensor, Y_tensor, Classifier_list,random_seed_num=21,FeatureSetName='Doc2Vec'):

    '''
    this funtion is aiming to perform the hyper-parameter tuning and feature selection

    Parameters
    ----------
    X_tensor:
            input features
    Y_tensor:
            input labels
    Classifier_list:
            list of classifier names that you want to use, should match the name in get_model() function
    random_seed_num:
            fix the random seed of classifier and oversampling method
    FeatureSetName:
            use to provide the name of current feature set name, as we may not want to apply feature selection for some
            specific feature set (e.g., like doc2vec)

    Return
    ----------
    Selected_cf:
            the classifier model after tuning the hyper-parameters
    Selected_fs
            the feature selection model
    best_scores
            the classification accuracy (within grid search) of best classifier model and best feature selection model
    '''
    pipeline_step_variance = 'variance_reduce'
    pipeline_step_smote = 'oversample'
    pipeline_step0 = 'StandardScaler'
    pipeline_step1 = 'reduce_dim'
    pipeline_step2 = 'classifier'
    num_of_feature=np.array(X_tensor).shape[1]

    pipe = Pipeline([
        (pipeline_step_variance, VarianceThreshold(threshold=(0.0000000001))),
        (pipeline_step_smote, SMOTE(random_state=random_seed_num)),
        (pipeline_step0, preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)),
        (pipeline_step1, SelectPercentile(feature_selection.f_classif)),
        (pipeline_step2, get_model(Classifier_list[0]))
    ])

    Selected_cf = []
    Selected_fs = []
    best_scores=[]

    for cf_item in Classifier_list:
        para_steps = {}
        if FeatureSetName == 'Doc2Vec':
            para_steps.update({pipeline_step1: [SelectPercentile(feature_selection.f_classif)],
                               pipeline_step1 + '__percentile': [100]
                               }
                              )
        else:
            para_steps.update({pipeline_step1: [SelectPercentile(feature_selection.f_classif)],
                               pipeline_step1 + '__percentile': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                               }
                              )
        #print(cf_item)
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
        else:
            print("Error: No such classifier")

        param_grid = para_steps

        skf = StratifiedKFold(n_splits=10,random_state=random_seed_num)
        loo = LeaveOneOut()
        grid = GridSearchCV(pipe, cv=skf, n_jobs=4, param_grid=param_grid, return_train_score="False")
        # print(X_tensor,Y_tensor)
        grid.fit(X_tensor, Y_tensor)
        # print(grid.cv_results_)
        # input()
        best_classifier, best_evaluator = grid.best_params_['classifier'], grid.best_params_['reduce_dim']

        Selected_cf.append(best_classifier)
        Selected_fs.append(best_evaluator)
        best_scores.append(grid.best_score_)

        #print(best_classifier)
        #print(best_evaluator)
        print(grid.best_score_, " ", cf_item)

    return Selected_cf, Selected_fs, best_scores



def get_model(name):
    '''
    This function is used to return classifier model with a classifier name as input
    '''

    random_seed_num=21
    if name == 'Logistic Regression':
        return LogisticRegression(max_iter=10000000,random_state=random_seed_num)
    elif name == 'SVM':
        return SVC(kernel='linear', probability=True, max_iter=10000000,random_state=random_seed_num)
    elif name == 'Decision tree':
        return tree.DecisionTreeClassifier(random_state=random_seed_num)
    elif name == 'SVC':
        return SVC(kernel='poly', probability=True, max_iter=10000000,random_state=random_seed_num)
    elif name == 'SVM_rbf':
        return SVC(kernel='rbf', probability=True, max_iter=10000000,random_state=random_seed_num)
    elif name == 'MultinomialNB':
        return MultinomialNB()
    elif name == 'Gradient Boosting':
        return GradientBoostingClassifier(n_estimators=100, max_features=1,random_state=random_seed_num)
    elif name == 'KNeighborsClassifier':
        return KNeighborsClassifier(3)
    elif name == 'MLP':
        return MLPClassifier(solver='lbfgs', max_iter=10000,random_state=random_seed_num)
    elif name == 'NaiveBayes':
        return GaussianNB()
    elif name == "AdaBoost":
        return AdaBoostClassifier(n_estimators=100,random_state=random_seed_num)
    elif name == "RandomForest":
        return RandomForestClassifier(max_depth=4, n_estimators=100, max_features=1,random_state=random_seed_num)
    else:
        raise ValueError('No such model')
