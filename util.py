#the aim of this code is to read the txt file and convert to a form which LIWC
#can directly use. Additionally, add the number of text into the excel log file

import os
import csv
import numpy as np

def label_features(file_address):
    labeled_data=[]
    with open(file_address, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:

            # check whether have label already
            if row[-1]=="Label":
                print("Info: Labeled file -",file_address)
                return

            if row[0]=="ID":
                labeled_data.append(row+["Label"])
            elif row[0].startswith("C") or row[0].startswith("S"):
                labeled_data.append(row+["Patient"])
            elif row[0].startswith("H"):
                labeled_data.append(row+["Healthy"])
            elif row[0].startswith("D"):
                labeled_data.append(row + ["Patient"])
            else:
                print("Error: False label")
    csvfile.close()

    with open(file_address,'w',newline="") as f:
        writer = csv.writer(f, delimiter=',')
        for item in labeled_data:
            #print(item)
            writer.writerow(item)
    f.close()
    print("Info: Has labeled the output file")

def write_list_to_csv(save_name,content_list):
    with open(save_name,'w',newline="") as f:
        writer = csv.writer(f, delimiter=',')
        for item in content_list:
            writer.writerow(item)
    f.close()

def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

def read_csv(file_address):
    file_contents = []
    with open(file_address, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            file_contents.append(row)
    return file_contents

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def check_file_exists(file_address):
    return os.path.isfile(file_address)


def write_result_three_class(feature_used, Acc, AUC_max, ConfuMatrix_max, save_file_address, ClfReport_,
                             baseline, classifier_name):
    # write result in to csv file
    if len(classifier_name) == 1:  # when tuning a single classifier
        classifier_name = classifier_name[0]
        with open(save_file_address, 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(
                [feature_used, 'D', ConfuMatrix_max[0], ConfuMatrix_max[1], ConfuMatrix_max[2], ClfReport_[0][0],
                 ClfReport_[1][0], ClfReport_[2][0], AUC_max, Acc, baseline, classifier_name])
            spamwriter.writerow(
                [feature_used, 'H', ConfuMatrix_max[3], ConfuMatrix_max[4], ConfuMatrix_max[5], ClfReport_[0][1],
                 ClfReport_[1][1], ClfReport_[2][1], AUC_max, Acc, baseline, classifier_name])
            spamwriter.writerow(
                [feature_used, 'S', ConfuMatrix_max[6], ConfuMatrix_max[7], ConfuMatrix_max[8], ClfReport_[0][2],
                 ClfReport_[1][2], ClfReport_[2][2], AUC_max, Acc, baseline, classifier_name])
    else:
        classifier_name = len(classifier_name)
        with open(save_file_address, 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(
                [feature_used, 'D', ConfuMatrix_max[0], ConfuMatrix_max[1], ConfuMatrix_max[2], ClfReport_[0][0],
                 ClfReport_[1][0], ClfReport_[2][0], AUC_max, Acc, baseline, classifier_name])
            spamwriter.writerow(
                [feature_used, 'H', ConfuMatrix_max[3], ConfuMatrix_max[4], ConfuMatrix_max[5], ClfReport_[0][1],
                 ClfReport_[1][1], ClfReport_[2][1], AUC_max, Acc, baseline, classifier_name])
            spamwriter.writerow(
                [feature_used, 'S', ConfuMatrix_max[6], ConfuMatrix_max[7], ConfuMatrix_max[8], ClfReport_[0][2],
                 ClfReport_[1][2], ClfReport_[2][2], AUC_max, Acc, baseline, classifier_name])


def write_title_three_class(save_file_address):
    # write result in to csv file
    if not os.path.exists(save_file_address):
        with open(save_file_address, 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(
                ['Feature', '', 'D', 'H', 'S', 'precision', "recall", "F-score", 'AUC', 'Acc', 'Baseline','Classifier'])


def write_result_two_class(feature_used, Acc, AUC_max, ConfuMatrix_max, save_file_address, ClfReport_,
                           baseline, classifier_name):
    # write result in to csv file
    if len(classifier_name) == 1:  # when tuning a single classifier
        classifier_name = classifier_name[0]
        with open(save_file_address, 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(
                [feature_used, 'patient', ConfuMatrix_max[0], ConfuMatrix_max[1], ClfReport_[0][0], ClfReport_[1][0],
                 ClfReport_[2][0], AUC_max, Acc, baseline,classifier_name])
            spamwriter.writerow(
                [feature_used, 'healthy', ConfuMatrix_max[2], ConfuMatrix_max[3], ClfReport_[0][1], ClfReport_[1][1],
                 ClfReport_[2][1], AUC_max, Acc, baseline,classifier_name])
    else:
        classifier_name = len(classifier_name)
        with open(save_file_address, 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(
                [feature_used, 'patient', ConfuMatrix_max[0], ConfuMatrix_max[1], ClfReport_[0][0], ClfReport_[1][0],
                 ClfReport_[2][0], AUC_max, Acc, baseline,classifier_name])
            spamwriter.writerow(
                [feature_used, 'healthy', ConfuMatrix_max[2], ConfuMatrix_max[3], ClfReport_[0][1], ClfReport_[1][1],
                 ClfReport_[2][1], AUC_max, Acc, baseline,classifier_name])


def write_title_two_class(save_file_address):
    # write result in to csv file
    if not os.path.exists(save_file_address):
        with open(save_file_address, 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(
                ['Feature', '', 'patient', 'healthy', 'precision', "recall", "F-score", 'AUC', 'Acc', 'Baseline',
                 'classifier'])


if __name__ == '__main__':
    print("This file provides utility functions")
