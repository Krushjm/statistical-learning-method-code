# encoding = utf-8

import numpy as np
import pandas as pd

from util.entropy import calc_info_gain, calc_info_gain_radio


class Tree:

    def __init__(self, feature=None, class_=None, node=[], data=[]):
        self.feature = feature
        self.class_ = class_
        self.node = node
        self.data = data

    def get_tree_info(self):
        if self.feature:
            print('feature: ', self.feature)
        if self.class_:
            print('class: ', self.class_)
        if self.node:
            print('node: ', self.node)
        if not data.empty:
            print('data: ', self.data)
        if self.node:
            for tree in self.node:
                tree.get_tree_info()

    def isLeaf(self):
        if self.class_:
            return True
        else:
            return False

    def get_leaf_number(self):
        if self.isLeaf():
            return 1
        else:
            return sum([sub_tree.get_leaf_number() for sub_tree in self.node])

    # def pruning(self, argu_a):
    #     if self.isLeaf():
    #         pass
    #     else:
    #         for sub_tree in self.node:
    #             sub_tree.pruning(argu_a)

def id3(dataset, label, threshold):
    X = dataset.iloc[:,:-1]
    y = dataset.iloc[:,-1]
    if np.unique(y).shape[0] == 1:
        return Tree(class_=np.unique(y)[0], data=dataset)
    elif not label:
        return Tree(class_=get_most_class(y), data=dataset)
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
                        ],
                        data=dataset)

def c4_5(dataset, label, threshold):
    X = dataset.iloc[:,:-1]
    y = dataset.iloc[:,-1]
    if np.unique(y).shape[0] == 1:
        return Tree(class_=np.unique(y)[0], data=dataset)
    elif not label:
        return Tree(class_=get_most_class(y), data=dataset)
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
                        ],
                        data=dataset)

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
    tree.get_tree_info()
    num = tree.get_leaf_number()
    print(num)
