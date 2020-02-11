import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings("ignore", category=DeprecationWarning)
from itertools import compress, product

import os, glob,csv
import statistics
from collections import Counter
import numpy as np
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix
from util import label_features
import pandas as pd
import pickle as pk
from util import write_result_two_class,write_title_two_class,write_result_three_class,write_title_three_class

def combinations(items):
    return (set(compress(items, mask)) for mask in product(*[[0, 1]] * len(items)))

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

def classifier_output_weight(Best_scores_pool,baseline):
    weights=[]
    baseline=0.5
    for item in Best_scores_pool:
        if item==1:
            item=0.999
        weight=0.5*np.log((1/baseline-1)*item/(1-item))
        weights.append(weight)

    return weights

def main():
    Classifier_list = ['Logistic Regression', 'SVM', 'Gradient Boosting', 'AdaBoost', 'RandomForest']
    imp_features_add = './result/Important_features.csv'

    Random_seed = 21  # fix random seed for a fully deterministically-reproducible run


    # # Load tmp files
    # Load feature set names
    file_addresses = glob.glob('./tmp/*pickle')

    predict_item_names=[]
    if file_addresses != []:
        for file_address in file_addresses:
            predict_item_names.append(os.path.basename(file_address).split(".")[0].split('_')[0])

    predict_item_name=sorted(set(predict_item_names))

    for i in range(len(predict_item_name)):
        if predict_item_name[i]=='CV':
            save_file_address='./result/prediction_results.csv'
        else:
            save_file_address ='./result/'+predict_item_name[i]+'_prediction_results.csv'

        file = [s for s in file_addresses if predict_item_name[i] in s]
        fr = open(file[0], 'rb')
        ML_temp_file = pk.load(fr)
        doc_label = ML_temp_file.doc_label

        counter = Counter(doc_label)
        baseline = counter.most_common(1)[0][1] / len(doc_label)

        if len(set(doc_label)) == 2:
            write_title_two_class(save_file_address)
        elif len(set(doc_label)) == 3:
            write_title_three_class(save_file_address)

        feature_set_name=[]
        cv_num=[]
        num_feature_sets=0
        if file_addresses != []:
            for file_address in file_addresses:
                feature_set_name.append(os.path.basename(file_address).split(".")[0].split('_')[-1])
                cv_num.append(int(os.path.basename(file_address).split(".")[0].split('_')[-2]))

            feature_set_names=list(set(feature_set_name))
            feature_set_names.sort()
            num_feature_sets=len(feature_set_names)
            cv_num=max(cv_num)
            print("Info: num_feature_sets is", num_feature_sets,feature_set_names,"No. CV loop is: ",cv_num)
        else:
            print("ERROR: No feature set input")
            quit()

        # # get the all the combinations of different feature sets
        # if num_feature_sets < 12:
        #     testing_list=list(combinations(range(num_feature_sets)))[1:]
        #     print("Info: posible combinations: ",len(testing_list))
        # else:
        #     print("ERROR: too many feature set, please change searching method.")
        #     quit()

        #calculate the classification results
        single_feature_sets=[[x] for x in range(0,len(feature_set_names))]
        Verbal=[[1,3,4,5]]
        NonVerbal=[[0,2,6]]
        All=[[0,1,2,3,4,5,6]]

        testing_list=single_feature_sets+Verbal+NonVerbal+All

        title_name=['Conver', 'Diction', 'DisVoice', 'Doc2Vec', 'LDA', 'LIWC', 'OpenSmile','Verbal','Nonverbal','All']
        Classifier_list = ['Logistic Regression', 'SVM', 'Gradient Boosting', 'AdaBoost', 'RandomForest','all classifier']

        start_loop=1
        acc=[]
        for clf in Classifier_list:
            acc_clf=[]
            for item in testing_list:
                #print(list(item))
                acc_=read_tmp_file(file_addresses,feature_set_names,list(item),doc_label,save_file_address,[clf],baseline,start_loop,len(testing_list)*len(Classifier_list),cv_num,title_name)
                start_loop=start_loop+1
                acc_clf.append(acc_)
            acc.append(acc_clf)

        df=pd.DataFrame(acc,index=Classifier_list,columns=title_name)
        df.to_csv('./result/test_all_clf_fs_results.csv')

def read_tmp_file(file_addresses,feature_set_names,feature_sets_id,doc_label,save_file_address,Classifier_list,baseline,start_loop,end_loop,cv_num,title_name):

    clf_acc_pool = []
    ConfuMatrix_pool = []
    ClfReport_pool = []
    AUC_pool, predict_label_pool, Important_features_pool, baseline_pool, num_top_pool = [], [], [], [], []

    scores = []
    predict_label = []
    real_label = []
    all_classifier_list = ['Logistic Regression', 'SVM', 'Gradient Boosting', 'AdaBoost', 'RandomForest']

    for loop in range(int(cv_num)+1):
        print("====CV: ",str(loop)," === Combination: ",str(start_loop),"/",str(end_loop),"==",str([feature_set_names[i] for i in feature_sets_id]),"========")
        files_in_the_loop=[s for s in file_addresses if '_'+str(loop)+'_' in s]

        Best_scores_pool = []
        pred_prob_pool = []


        #print("feature set names: ", [feature_set_names[i] for i in feature_sets_id])
        for id in feature_sets_id:

            file = [s for s in files_in_the_loop if '_' + feature_set_names[id] in s]
            fr=open(file[0],'rb')
            ML_temp_file=pk.load(fr)
            Best_scores=ML_temp_file.best_score_in_grid_search[0]
            pred_prob=ML_temp_file.clf_scores[0]

            if not Classifier_list==['all classifier']:

                classifier_id=[i for i,x in enumerate(all_classifier_list) if x in Classifier_list]
                print([Best_scores[x] for x in classifier_id],[pred_prob[x] for x in classifier_id])

                Best_scores_pool = Best_scores_pool + [Best_scores[x] for x in classifier_id]
                pred_prob_pool = pred_prob_pool + [pred_prob[x] for x in classifier_id]

            else:

                Best_scores_pool = Best_scores_pool + Best_scores
                pred_prob_pool = pred_prob_pool + pred_prob

        voting_weight = classifier_output_weight(Best_scores_pool, baseline)
        pred_prob_total = pred_prob_pool

        # get comprehensive prediction scores and its predictive class
        pred_score = np.array(pred_prob_total[0])
        pred_score.fill(0)
        for item in range(len(voting_weight)):
            add = voting_weight[item] * np.array(pred_prob_total[item])
            pred_score = pred_score + add

        pred_score = pred_score / pred_score.sum(axis=1, keepdims=1)
        pred = [list(sorted(set(doc_label)))[item.index(max(item))] for item in pred_score.tolist()]

        print('True: ', ML_temp_file.true_label, 'Predict: ', pred)
        print(pred_score.tolist())

        scores=scores+pred_score.tolist()
        predict_label=predict_label+pred
        real_label=real_label+ML_temp_file.true_label

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
    print("classification accuracy:", acc_result)
    print("Confusion Matrix: ", CM)
    print(CR_print)
    print(CR)
    print(auc1)

    return acc_result

if __name__ == "__main__":
    main()

