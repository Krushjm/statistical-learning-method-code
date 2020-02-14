# encoding = utf-8

import numpy as np
import pandas as pd

from util.entropy import calc_info_gain, calc_info_gain_radio


class Tree:
    
    def __init__(self, feature=None, class_=None, node=[]):
        self.feature = feature
        self.class_ = class_
        self.node = node
        
    def get_tree_info(self):
        print(self.feature)
        print(self.class_)
        print(self.node)
        if self.node:
            for tree in self.node:
                tree.get_tree_info()


def id3(dataset, label, threshold):
    X = dataset.iloc[:,:-1]
    y = dataset.iloc[:,-1]
    if np.unique(y).shape[0] == 1:
        return Tree(class_=np.unique(y)[0])
    elif not label:
        return Tree(class_=get_most_class(y))
    else:
        feature, info_gain = calc_info_gain(dataset, label)
        if info_gain < threshold:
            return Tree(class_=get_most_class(y))
        else:
            label.remove(feature)
            print(label)
            return Tree(feature=feature,
                        node=[
                            id3(dataset[dataset[feature] == value], label, threshold) for value in np.unique(dataset[feature])
                        ])
            
def c4_5(dataset, label, threshold):
    X = dataset.iloc[:,:-1]
    y = dataset.iloc[:,-1]
    if np.unique(y).shape[0] == 1:
        return Tree(class_=np.unique(y)[0])
    elif not label:
        return Tree(class_=get_most_class(y))
    else:
        feature, info_gain = calc_info_gain_radio(dataset, label)
        if info_gain < threshold:
            return Tree(class_=get_most_class(y))
        else:
            label.remove(feature)
            print(label)
            return Tree(feature=feature,
                        node=[
                            c4_5(dataset[dataset[feature] == value], label, threshold) for value in np.unique(dataset[feature])
                        ])
    
def get_most_class(y):
    class_frequency = np.array(np.unique(y, return_counts=True)).T
    most_class = class_frequency[class_frequency[:, -1].argsort()][-1,0]
    return most_class


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
    # tree = id3(data, labels[:-1], 0)
    tree = c4_5(data, labels[:-1], 0)
