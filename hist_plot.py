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
from util import label_features,write_list_to_csv
import pandas as pd
import pickle as pk
import time
from datetime import timedelta
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from util import write_result_two_class,write_title_two_class,write_result_three_class,write_title_three_class
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import stats

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
    Classifier_list = ['Logistic Regression', 'SVM', 'Gradient Boosting', 'AdaBoost', 'RandomForest']
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
        doc_label_ = list_[0]["Label"].loc[common_file_names].tolist()

        # Identify non-numeric label
        if ~doc_label_[0].isdigit():
            le = preprocessing.LabelEncoder()
            le.fit(doc_label_)
            doc_label = le.transform(doc_label_)

        print("Info: number of file is: ", len(doc_label))

        # save features into LoadFeatures_pool
        for df_item in list_:
            df_item.drop(["Label"], axis=1, inplace=True)
            LoadFeatures_pool.append(df_item.loc[common_file_names])

    elif len(FeatureSetName_pool) == 1:
        common=list_[0].index

        # get the samples ID and labels
        common_file_names = common.tolist()
        doc_label_ = list_[0]["Label"].loc[common_file_names].tolist()

        # Identify non-numeric label
        if ~doc_label_[0].isdigit():
            le = preprocessing.LabelEncoder()
            le.fit(doc_label_)
            doc_label = le.transform(doc_label_)

        print("Info: number of file is: ", len(doc_label))

        # save features into LoadFeatures_pool
        list_[0].drop(["Label"], axis=1, inplace=True)
        LoadFeatures_pool.append(list_[0])

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

    # LoadFeatures_pool: features

    # LoadFeatureNames_pool: feature names

    # FeatureSetName_pool: feature set names

    # doc_label: labels

    # save all features

    df_new=pd.concat(LoadFeatures_pool, axis=1)
    df_new['Label']=doc_label
    df_new.to_csv('./IMH_all_features.csv')

    if len(FeatureSetName_pool) > 1:
        for i in range(len(LoadFeatures_pool)):
            LoadFeatures_pool[i]=LoadFeatures_pool[i].values

    # create new list
    LoadFeatureNames_pool_filtered = []
    LoadFeatures_pool_filtered = []
    FeatureSetName_pool_filtered=[]

    LoadFeatureNames_pool_save = []
    Features_pValue_pool_save = []
    FeatureSetName_pool_save=[]

    for FeaSet_id in range(len(FeatureSetName_pool)):
        if FeatureSetName_pool[FeaSet_id] in ['Diction','LIWC','LDA','Doc2Vec','OpenSmile','DisVoice','Conver','Movement','Affectiva','Opsis','OpenFace']:
            features_name=[]
            features=[]

            features_name_save=[]
            features_pValue_save=[]
            for Fea_id in range(len(LoadFeatureNames_pool[FeaSet_id])):
                class_a = LoadFeatures_pool[FeaSet_id][:, Fea_id][doc_label == 0]
                class_b = LoadFeatures_pool[FeaSet_id][:, Fea_id][doc_label == 1]
                try:
                    stat, p = stats.kruskal(class_a, class_b)
                except:
                    p = 1.0
                # print(FeatureSetName_pool[FeaSet_id],LoadFeatureNames_pool[FeaSet_id][Fea_id],stat, p,Fea_id)
                # interpret
                alpha = 0.9
                if p < alpha:
                    features_name.append(LoadFeatureNames_pool[FeaSet_id][Fea_id])
                    features.append(LoadFeatures_pool[FeaSet_id][:,Fea_id].tolist())

                # save all p values
                features_name_save.append(LoadFeatureNames_pool[FeaSet_id][Fea_id])
                features_pValue_save.append(p)

            print(FeatureSetName_pool[FeaSet_id],len(features_name))

            LoadFeatureNames_pool_filtered.append(features_name)
            LoadFeatures_pool_filtered.append(np.array(features).T)
            FeatureSetName_pool_filtered.append(FeatureSetName_pool[FeaSet_id])

            LoadFeatureNames_pool_save.append(features_name_save)
            Features_pValue_pool_save.append(features_pValue_save)
            FeatureSetName_pool_save.append(FeatureSetName_pool[FeaSet_id])

    # save top 10 p-values of each data set

    # read task abbreviation
    counter_ = Counter(doc_label_)
    task_abbreviation = ''
    for item in counter_.keys():
        print(item)
        task_abbreviation =task_abbreviation + item[0]
    print('task_abbreviation: ',task_abbreviation)

    df=[]
    for FeaSet_id in range(len(FeatureSetName_pool_save)):
        data_frame_header=[[FeatureSetName_pool_save[FeaSet_id],FeatureSetName_pool_save[FeaSet_id]]]

        fn_pvalue = []
        rank=np.argsort(Features_pValue_pool_save[FeaSet_id])
        for i in range(10):
            fn_pvalue.append([LoadFeatureNames_pool_save[FeaSet_id][rank[i]],Features_pValue_pool_save[FeaSet_id][rank[i]]])

        df.append(pd.DataFrame(data_frame_header+fn_pvalue,index=None,columns=None))

    df_all=pd.concat(df,axis=1,sort=False)
    df_all.to_csv('result/Feature_P_ranking_'+task_abbreviation+'.csv',index=None,columns=None,header=None)

    save_directory_dis = './histogram'
    if not os.path.exists(save_directory_dis):
        os.makedirs(save_directory_dis)

    # print the histogram
    if len(counter_.keys())==2:
        plot_his_2Class(p_value_list_path='./result/Feature_P_ranking_' + task_abbreviation + '.csv',
                        entire_features_path='./IMH_all_features.csv', abbr=task_abbreviation)
    elif len(counter_.keys())==3:
        plot_his_2Class(p_value_list_path='./result/Feature_P_ranking_' + task_abbreviation + '.csv',
                        entire_features_path='./IMH_all_features.csv', abbr=task_abbreviation)
    else:
        print('Error: number of class is incorrect.')

    # plot the distribution of features




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

def plot_his_2Class(p_value_list_path,entire_features_path,abbr):

    salient_features_list = pd.read_csv(p_value_list_path, header=None, index_col=None)
    features_list = pd.read_csv(entire_features_path, header=0, index_col=0)

    save_directory = './histogram'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # select feature set
    for j in range(0, int(len(salient_features_list.columns)/2)):

        plot_fig_num = 10
        for i in range(plot_fig_num):
            print('plotting: (',j,',',i,')')
            feature_name = salient_features_list[j * 2][i + 1]

            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

            label = features_list['Label']
            df = features_list[feature_name]

            filter_col_A = [location for location, label_ in enumerate(label.values.tolist()) if label_ == 0]
            sampleA = df.loc[df.index[filter_col_A]].values.tolist()

            filter_col_B = [location for location, label_ in enumerate(label.values.tolist()) if label_ == 1]
            sampleB = df.loc[df.index[filter_col_B]].values.tolist()

            all_data = [sampleA, sampleB]
            # plot violin plot
            axes.violinplot(all_data,
                            showmeans=False,
                            showmedians=True)
            axes.set_title(feature_name)

            # adding horizontal grid lines
            for ax in [axes]:
                ax.yaxis.grid(True)
                ax.set_xticks([y + 1 for y in range(len(all_data))])
                # ax.set_xlabel('xlabel')
                # ax.set_ylabel('ylabel')

            # add x-tick labels
            plt.setp(axes, xticks=[y + 1 for y in range(len(all_data))],
                     xticklabels=[abbr[0], abbr[1]])
            fig.savefig(save_directory+'/'+abbr+'_'+salient_features_list[j * 2][0]+'_'+feature_name+'.png', dpi=fig.dpi)


def plot_his_3Class(p_value_list_path,entire_features_path,abbr):

    salient_features_list = pd.read_csv(p_value_list_path, header=None, index_col=None)
    features_list = pd.read_csv(entire_features_path, header=0, index_col=0)

    save_directory='./histogram'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # select feature set
    for j in range(0, len(salient_features_list.columns)):

        plot_fig_num = 10
        for i in range(plot_fig_num):

            feature_name = salient_features_list[j * 2][i + 1]

            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

            label = features_list['Label']
            df = features_list[feature_name]

            filter_col_A = [location for location, label_ in enumerate(label.values.tolist()) if label_ == 0]
            sampleA = df.loc[df.index[filter_col_A]].values.tolist()

            filter_col_B = [location for location, label_ in enumerate(label.values.tolist()) if label_ == 1]
            sampleB = df.loc[df.index[filter_col_B]].values.tolist()

            filter_col_C = [location for location, label_ in enumerate(label.values.tolist()) if label_ == 2]
            sampleC = df.loc[df.index[filter_col_C]].values.tolist()

            all_data = [sampleA, sampleB, sampleC]
            # plot violin plot
            axes.violinplot(all_data,
                            showmeans=False,
                            showmedians=True)
            axes.set_title(feature_name)

            # adding horizontal grid lines
            for ax in [axes]:
                ax.yaxis.grid(True)
                ax.set_xticks([y + 1 for y in range(len(all_data))])

            # add x-tick labels
            plt.setp(axes, xticks=[y + 1 for y in range(len(all_data))],
                     xticklabels=[abbr[0], abbr[1], abbr[2]])
            fig.savefig(save_directory+'/'+abbr+'_'+salient_features_list[j * 2][0]+'_'+feature_name+'.png', dpi=fig.dpi)

if __name__ == "__main__":
    main()