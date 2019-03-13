import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os, glob,csv
import statistics
from collections import Counter
from sklearn_classifier import open_file, get_model, CV_Para_selection
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectPercentile
from imblearn.pipeline import Pipeline
from sklearn import feature_selection
from util import label_features
import pandas as pd
import pickle as pk
import time
from datetime import timedelta
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from util import write_result_two_class,write_title_two_class,write_result_three_class,write_title_three_class

class ML_temp_file(object):
    '''
    This class is aiming to save the necessary metrics and information in each cross validation loop.
    The saved information could used to reproduce the experimental results, and test accuracies of combinations of
    different feature sets.
    '''

    def __init__(self, cv_num, feature_set_name,clf_model, fs_model, best_score_in_grid_search, clf_scores, important_features,
                 important_features_name,doc_label,true_label):
        self.cv_num=cv_num
        self.feature_set_name=feature_set_name
        self.clf_model = clf_model  # classification model
        self.fs_model = fs_model  # feature selection model
        self.best_score_in_grid_search = best_score_in_grid_search
        self.clf_scores = clf_scores
        self.important_features = important_features
        self.important_features_name = important_features_name
        self.doc_label=doc_label
        self.true_label=true_label

    def to_pickle(self,path):
        fp=open(path,"wb")
        pk.dump(self,fp)

def main():
    Classifier_list = [ 'Gradient Boosting']
    save_file_address = './result/Audio_cf_results.csv'
    imp_features_add = './result/Important_features.csv'

    Random_seed = 21  # fix random seed for a fully deterministically-reproducible run

    # Load feature sets and label the data set
    # If no 'Label' column, I label samples based on the sample ID, see details in label_features function
    file_addresses = glob.glob('./features/*csv')
    if file_addresses != []:
        for file_address in file_addresses:
            label_features(file_address)
    else:
        print("ERROR: No feature set input")

    # load feature sets in csv files
    list_ = []
    LoadFeatures_pool = []
    FeatureSetName_pool = []
    LoadFeatureNames_pool=[]
    for file_ in file_addresses:
        FeatureSetName = os.path.basename(file_).split("_")[0]
        df = pd.read_csv(file_, index_col=0, header=0)
        list_.append(df)
        FeatureSetName_pool.append(FeatureSetName)
        LoadFeatureNames=df.columns.tolist()[0:-1]
        LoadFeatureNames_pool.append(LoadFeatureNames)
        print("Info: Got the " + FeatureSetName + " Feature Set")

    # select the same items in multiple data sets
    common = list_[0].index
    if len(FeatureSetName_pool) > 1:
        for df_item in list_[1:]:
            common = common[common.isin(df_item.index)]

        # get the samples ID and labels
        common_file_names = common.tolist()
        doc_label = list_[0]["Label"].loc[common_file_names].tolist()

        # Identify non-numeric label
        if ~doc_label[0].isdigit():
            le = preprocessing.LabelEncoder()
            le.fit(doc_label)
            doc_label = le.transform(doc_label)

        print("Info: number of file is: ", len(doc_label))

        # save features into LoadFeatures_pool
        for df_item in list_:
            df_item.drop(["Label"], axis=1, inplace=True)
            LoadFeatures_pool.append(df_item.loc[common_file_names].values.tolist())

    elif len(FeatureSetName_pool) == 1:
        common=list_[0].index

        # get the samples ID and labels
        common_file_names = common.tolist()
        doc_label = list_[0]["Label"].loc[common_file_names].tolist()

        # Identify non-numeric label
        if ~doc_label[0].isdigit():
            le = preprocessing.LabelEncoder()
            le.fit(doc_label)
            doc_label = le.transform(doc_label)

        print("Info: number of file is: ", len(doc_label))

        # save features into LoadFeatures_pool
        list_[0].drop(["Label"], axis=1, inplace=True)
        LoadFeatures_pool.append(list_[0].values.tolist())

    if len(set(doc_label)) == 2:
        write_title_two_class(save_file_address)
    elif len(set(doc_label)) == 3:
        write_title_three_class(save_file_address)
    else:
        print("ERROR: too many class, please modify the output function.")
        quit()

    # calculate the baseline accuracy based on majority class
    counter = Counter(doc_label)
    baseline = counter.most_common(1)[0][1] / len(doc_label)

    # save the important feature counts in each LOOCV fold
    selected_features_counting_pool=[]
    selected_features_counting_num_pool=[]
    for Num_FeatureSet in range(len(FeatureSetName_pool)):
        selected_features_counting = np.zeros(len(LoadFeatureNames_pool[Num_FeatureSet]))
        selected_features_counting_num = 0
        selected_features_counting_pool.append(selected_features_counting)
        selected_features_counting_num_pool.append(selected_features_counting_num)

    scores = []
    predict_label = []
    real_label=[]
    loop=0

    skf=StratifiedKFold(n_splits=10)
    for train_index, test_index in skf.split(range(len(LoadFeatures_pool[0])),doc_label):
    # loo=LeaveOneOut()
    # for train_index, test_index in loo.split(range(len(LoadFeatures_pool[0]))):
        start_time = time.time()
        print(
            "===============CV=" + str(loop) + "==================")

        Best_scores_pool = []
        pred_prob_pool = []


        for Num_FeatureSet in range(len(FeatureSetName_pool)):

            # create tmp file
            inter_var = ML_temp_file(cv_num=loop, feature_set_name=FeatureSetName_pool[Num_FeatureSet], clf_model=[],
                                     fs_model=[], best_score_in_grid_search=[],
                                     clf_scores=[], important_features=[],
                                     important_features_name=[], doc_label=doc_label,true_label=[doc_label[x] for x in test_index])

            print('-----------' + FeatureSetName_pool[Num_FeatureSet] + '--------------')
            X_train = [LoadFeatures_pool[Num_FeatureSet][x] for x in train_index]
            Y_train = [doc_label[x] for x in train_index]
            X_test = [LoadFeatures_pool[Num_FeatureSet][x] for x in test_index]
            Y_Test = [doc_label[x] for x in test_index]
            Best_scores, pred_prob, Important_features,inter_var = Ensemble_Prediction(X_train, Y_train, X_test, Y_Test,
                                                         Classifier_list, inter_var,Random_seed,FeatureSetName_pool[Num_FeatureSet])

            # save best scores in CV grid search and the prediction scores of each classifier
            Best_scores_pool = Best_scores_pool + Best_scores
            pred_prob_pool = pred_prob_pool + pred_prob

            # # get selected features
            # get how many time current feature set was used
            selected_features_counting_num=selected_features_counting_num_pool[Num_FeatureSet]
            # get the feature selection number of current feature set
            selected_features_counting=selected_features_counting_pool[Num_FeatureSet]
            for item in Important_features:
                selected_features_counting_num = selected_features_counting_num + 1
                selected_features_counting[item] = selected_features_counting[item] + 1
            selected_features_counting_pool[Num_FeatureSet]=selected_features_counting
            selected_features_counting_num_pool[Num_FeatureSet]=selected_features_counting_num

            # save information to tmp file
            inter_var.clf_scores.append(pred_prob)
            inter_var.important_features_name.append(LoadFeatureNames_pool[Num_FeatureSet])
            save_tmp_file_add='./tmp/CV_'+str(loop)+'_'+str(inter_var.feature_set_name)+'.pickle'
            inter_var.to_pickle(save_tmp_file_add)

        # weight different classifiers
        voting_weight = classifier_output_weight(Best_scores_pool, baseline)
        pred_prob_total = pred_prob_pool

        # get comprehensive prediction scores and its predictive class
        pred_score = np.array(pred_prob_total[0])
        pred_score.fill(0)
        for item in range(len(voting_weight)):
            add = voting_weight[item] * np.array(pred_prob_total[item])
            pred_score = pred_score + add

        pred_score = pred_score / pred_score.sum(axis=1, keepdims=1)
        pred = [list(set(doc_label))[item.index(max(item))] for item in pred_score.tolist()]
        end_time = time.time()
        print('True: ', [doc_label[x] for x in test_index], 'Predict: ', pred, " time remine:", str(timedelta(seconds=(len(doc_label)-loop-1)*(end_time-start_time))))
        #print(pred_score.tolist())
        # input()

        scores=scores+pred_score.tolist()
        predict_label=predict_label+pred
        real_label=real_label+[doc_label[x] for x in test_index]
        loop=loop+1

    predict_label = np.array(predict_label)
    acc_result = accuracy_score(real_label, predict_label)
    if len(set(predict_label)) == 2:
        scores = np.array(scores)
        # print(scores)
        fpr, tpr, thresholds = metrics.roc_curve(real_label, scores[:, 1])
        auc = metrics.auc(fpr, tpr)
        auc1 = max(auc, 1 - auc)
    elif len(set(predict_label)) == 3:
        auc1 = 0
    CM = confusion_matrix(real_label, predict_label).ravel()
    CR_print = classification_report(real_label, predict_label)
    CR = precision_recall_fscore_support(real_label, predict_label)
    # baseline=max((CM[0]+CM[1])/(CM[0]+CM[1]+CM[2]+CM[3]),(CM[2]+CM[3])/(CM[0]+CM[1]+CM[2]+CM[3]))
    print("classification accuracy:", acc_result)
    print("Confusion Matrix: ", CM)
    print(CR_print)
    print(CR)
    print(auc1)

    feature_countings_rank=[]
    feature_names_rank=[]
    for item in range(len(selected_features_counting_num_pool)):
        # calculate the most salient features
        features_rank_=np.argsort(-selected_features_counting_pool[item])
        feature_countings_rank=feature_countings_rank+list(selected_features_counting_pool[item][features_rank_]/selected_features_counting_num_pool[item])
        feature_names_rank=feature_names_rank+list(LoadFeatureNames_pool[item][i] for i in features_rank_)

    features_rank_=np.argsort(-np.array(feature_countings_rank))
    feature_countings_rank=np.array(feature_countings_rank)
    feature_countings_rank=feature_countings_rank[features_rank_]
    feature_names_rank=list(feature_names_rank[i] for i in features_rank_)

    # write result in to csv file
    with open(imp_features_add, 'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(feature_names_rank)
        spamwriter.writerow(feature_countings_rank)

    # save all features into
    feature_used = 'All Featrues'
    if len(set(doc_label)) == 2:
        write_result_two_class(feature_used, acc_result, auc1, CM, save_file_address, CR, baseline,Classifier_list)
    elif len(set(doc_label)) == 3:
        write_result_three_class(feature_used, acc_result, auc1, CM, save_file_address, CR, baseline,Classifier_list)
    else:
        print("ERROR: too many class, please modify the output function.")


def Ensemble_Prediction(X_train, Y_train, X_test, Y_test, Classifier_list, inter_var,
                            Random_seed=21, FeatureSetName='Doc2Vec'):
    '''
    This function is used to weight different classifiers based on the output of cross-validation grid search

    Parameters
    ----------
    X_train:
        the training data (list or numpy array)
    Y_train:
        label of training data (list or numpy array)
    X_test:
        the testing data (list or numpy array)
    Y_test:
        label of testing data (list or numpy array)
    Classifier_list:
        list of classifier names that you want to use, should match the name in get_model() function
    inter_var:
        ML_temp_file class file. Use to save the necessary information
    Random_seed:
        fix the random seed of classifier and oversampling method
    FeatureSetName:
        use to provide the name of current feature set name, as we may not want to apply feature selection for some
        specific feature set (e.g., like doc2vec)

    Return
    ----------
    weights: the weight of each classifier

    '''

    # applied cross-validation grid search
    #model_cf, model_fs, best_scores = CV_Para_selection(X_train, Y_train, Classifier_list,Random_seed,FeatureSetName)

    # following code is used to manually determine the classifier and feature selection model
    model_cf=[]
    model_fs=[]
    best_scores=[]
    for model in Classifier_list:
        model_cf.append(get_model(model))
        model_fs.append(SelectPercentile(feature_selection.f_classif, percentile=100))
        best_scores.append(0.6) # give weights to different classifiers

    inter_var.clf_model.append(model_cf)
    inter_var.fs_model.append(model_fs)
    inter_var.best_score_in_grid_search.append(best_scores)

    predict_scores=[]
    Important_features=[]
    if len(model_cf) == len(model_fs):
        #print("Info: predicting the testing data, feature len:", len(model_cf))
        for id in range(0, len(model_cf)):
            clf_fs = model_fs[id]  # feature selection
            clf_classifier = model_cf[id]  # classifier model

            pipeline_step_variance = 'variance_reduce'
            pipeline_step_smote = 'oversample'
            pipeline_step0 = 'StandardScaler'
            pipeline_step1 = 'reduce_dim'
            pipeline_step2 = 'classifier'
            # num_of_feature=X_tensor.shape[1]

            pipe = Pipeline([
                (pipeline_step_variance, VarianceThreshold(threshold=(0.0000000001))),
                (pipeline_step_smote, SMOTE(random_state=Random_seed)),
                (pipeline_step0, preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)),
                (pipeline_step1, clf_fs),
                (pipeline_step2, clf_classifier)
            ])

            pipe.fit(X_train, Y_train)
            score = pipe.predict_proba(X_test)
            pred_class = pipe.predict(X_test)

            print("True:", Y_test, "Predict:", pred_class, 'Classifier:', Classifier_list[id])

            del_small_var=pipe.named_steps[pipeline_step_variance].get_support(indices=True)
            Important_features.append(del_small_var[pipe.named_steps[pipeline_step1].get_support(indices=True)])
            predict_scores.append(score.tolist())

    #pred = [avg_score.tolist().index(max(avg_score))]
    return best_scores,predict_scores,Important_features,inter_var

def classifier_output_weight(Best_scores_pool,baseline):
    '''
    This function is used to weight different classifiers based on the output of cross-validation grid search

    Parameters
    ----------
    Best_scores_pool: the best cross-validation grid search accuracies of each input model
    baseline: the majority voting accuracy

    Return
    ----------
    weights: the weight of each classifier

    '''
    weights=[]
    for item in Best_scores_pool:
        weight=0.5*np.log((1/baseline-1)*item/(1-item))
        weights.append(weight)

    return weights


if __name__ == "__main__":
    main()

