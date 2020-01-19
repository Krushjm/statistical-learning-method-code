# example 3.2

import math
from operator import itemgetter

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


class kD_Node:

    def __init__(self, data_point, depth, left, right):
        self.data_point = data_point
        self.depth = depth
        self.left = left
        self.right = right


class kD_Tree:

    def __init__(self, data):
        self.d = data
        self.k = len(data[0])
        self.node = self._construct_tree(self.d)

    def _construct_tree(self, data, depth=0):
        if len(data) == 1:
            return self._create_node(data[0], depth, None, None)
        elif len(data) > 1:
            left_data, data_point, right_data = self._split_data(data, depth)
            left_node = self._construct_tree(left_data, depth + 1)
            right_node = self._construct_tree(right_data, depth + 1)
            return self._create_node(data_point, depth, left_node, right_node)
        else:
            pass

    def _split_data(self, data, depth):
        split_axis = depth % self.k
        sorted_data = sorted(data, key=itemgetter(split_axis))
        median = int(len(sorted_data) / 2)
        left_slice = sorted_data[:median]
        median_point = sorted_data[median]
        right_slice = sorted_data[median+1:]
        return left_slice, median_point, right_slice

    def _create_node(self, data_point, depth, left_node, right_node):
        return kD_Node(data_point, depth, left_node, right_node)

    def pre_order_traverse(self, node=None, depth=0):
        if depth == 0 and node is None:
            node = self.node
        if node:
            # for i in range(0, depth):
            #     print('\t', end='')
            # print(node.data_point)
            self.pre_order_traverse(node.left, depth + 1)
            self.pre_order_traverse(node.right, depth + 1)
        else:
            pass

    # example 3.3
    def search_nearest(self, node, x):
        target = np.array([])
        current = node.data_point
        axis = node.depth % self.k
        if x[axis] < current[axis]:
            if node.left:
                target = self.search_nearest(node.left, x)
                if node.right and self._distance(x, current, depth=node.depth) < self._distance(x, target):
                    target = self.search_nearest(node.right, x)
        else:
            if node.right:
                target = self.search_nearest(node.right, x)
                if node.left and self._distance(x, current, depth=node.depth) < self._distance(x, target):
                    target = self.search_nearest(node.left, x)
        if target.size:
            if self._distance(x, current) < self._distance(x, target):
                return current
            else:
                return target
        else:
            return current

    def _distance(self, target_x, node_x, depth=None):
        x1 = np.array(target_x)
        x2 = np.array(node_x)
        if depth:
            axis = depth % self.k
            x1 = np.delete(x1, axis)
            x2 = np.delete(x2, axis)
        distance = np.linalg.norm(x1 - x2)
        return distance



if __name__ == "__main__":
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['label'] = iris.target
    iris_df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(iris_df.iloc[:100, [0, 1, -1]])
    X, y = data[:,:-1], data[:,-1]
    unique_index = np.unique(X, axis=0, return_index=True)
    X, y = X[unique_index[1]], y[unique_index[1]]
    kd_tree = kD_Tree(X)
    kd_tree.pre_order_traverse()
    print(kd_tree.search_nearest(kd_tree.node, np.array([4.9, 2.9])))
