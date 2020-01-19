import math

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


class GaussianBayesianClassify:
    
    def __init__(self):
        pass
    
    def fit(self, X, y, x):
        if x[0] not in X:
            print('x is not in dataset!')
            return
        p_y = dict()
        p_x_y = dict()
        for val in np.unique(y):
            p_y[val] = np.sum(y == val) / y.shape[0]
            # print(val, p_y[val])
            for val2 in np.unique(X[:,0]):
                p_x_y[(val2, val)] = np.sum(X[:,0] == val2) * p_y[val] / np.sum(y == val)
                # print('col1', (val2, val), p_x_y[(val2, val)])
            for val3 in np.unique(X[:,1]):
                p_x_y[(val3, val)] = np.sum(X[:,1] == val3) * p_y[val] / np.sum(y == val)
                # print('col2', (val3, val), p_x_y[(val3, val)])
        result = {val: p_y[val] * p_x_y[(x[0], val)] * p_x_y[(x[1], val)] for val in p_y.keys()}
        prob, y1 = max(zip(result.values(), result.keys()))
        print((y1, float('%.5f' % prob)))
    
    def score(self):
        pass
    

if __name__ == "__main__":
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    X, y = data[:,:-1], data[:,-1]
    classify = GaussianBayesianClassify()
    classify.fit(X, y, np.array([5.2, 3.1]))