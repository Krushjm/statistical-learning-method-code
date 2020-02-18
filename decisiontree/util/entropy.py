from math import log2

import numpy as np
import pandas as pd


def calc_entropy(dataset):
    dataset_count = dataset.shape[0]  # |D|
    entropy = 0.0
    y_counts = {y: dataset[dataset.iloc[:,-1] == y].shape[0] for y in set(dataset.iloc[:,-1])}  # |C_k|
    entropy = sum([-(y_count / dataset_count) * log2(y_count / dataset_count) for y_count in y_counts.values()])
    return round(entropy, 3)

def calc_conditional_entropy(dataset, feature_name):
    dataset_count = dataset.shape[0]  # |D|
    conditional_entropy = 0.0
    for value in np.unique(dataset[feature_name]):
        sub_dataset = dataset[dataset[feature_name] == value]  # D_i
        value_count = sub_dataset.shape[0]  # |D_i|
        y_counts = {y: sub_dataset[sub_dataset.iloc[:,-1] == y].shape[0] for y in set(sub_dataset.iloc[:,-1])}  # |D_ik|
        conditional_entropy += sum([-(value_count / dataset_count) * (y_count / value_count) * log2(y_count / value_count) for y_count in y_counts.values()])
    return round(conditional_entropy, 3)

def calc_feature_entropy(dataset, feature_name):
    dataset_count = dataset.shape[0]
    feature_entropy = 0.0
    for value in np.unique(dataset[feature_name]):
        sub_dataset = dataset[dataset[feature_name] == value]
        value_count = sub_dataset.shape[0]
        feature_entropy += -(value_count / dataset_count) * log2(value_count / dataset_count)
    return round(feature_entropy, 3)

def calc_expirical_entropy():
    pass

def calc_info_gain(dataset, feature_list):
    entropy = calc_entropy(dataset)
    info_gain = 0.0
    best_feature = None
    for feature_name in feature_list:
        current_info_gain = entropy - calc_conditional_entropy(dataset, feature_name)
        # print(feature_name, round(current_info_gain, 3))
        if current_info_gain > info_gain:
            info_gain = current_info_gain
            best_feature = feature_name
    return best_feature, info_gain

def calc_info_gain_radio(dataset, feature_list):
    entropy = calc_entropy(dataset)
    info_gain_radio = 0.0
    best_feature = None
    for feature_name in feature_list:
        current_info_gain_radio = (entropy - calc_conditional_entropy(dataset, feature_name)) / calc_feature_entropy(dataset, feature_name)
        # print(feature_name, round(current_info_gain_radio, 3))
        if current_info_gain_radio > info_gain_radio:
            info_gain_radio = current_info_gain_radio
            best_feature = feature_name
    return best_feature, info_gain_radio


if __name__ == "__main__":
    datasets = [
        ['青年', '否', '否', '一般', '否'],
        ['青年', '否', '否', '好', '否'],
        ['青年', '是', '否', '好', '是'],
        ['青年', '是', '是', '一般', '是'],
        ['青年', '否', '否', '一般', '否'],
        ['中年', '否', '否', '一般', '否'],
        ['中年', '否', '否', '好', '否'],
        ['中年', '是', '是', '好', '是'],
        ['中年', '否', '是', '非常好', '是'],
        ['中年', '否', '是', '非常好', '是'],
        ['老年', '否', '是', '非常好', '是'],
        ['老年', '否', '是', '好', '是'],
        ['老年', '是', '否', '好', '是'],
        ['老年', '是', '否', '非常好', '是'],
        ['老年', '否', '否', '一般', '否'],
    ]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况', '类别']
    data = pd.DataFrame(datasets, columns=labels)
    # calc_conditional_entropy(data, '信贷情况')
    calc_info_gain(data, labels[:-1])
