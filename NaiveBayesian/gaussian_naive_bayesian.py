import math

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


class GaussianBayesianClassify:

    def __init__(self):
        pass

    def _maximum_likelihood_estimation(self, X, y):
        p_y, p_x_y = dict(), dict()
        for val_y in np.unique(y):
            p_y[val_y] = np.sum(y == val_y) / y.shape[0]
            for col in range(0, X.shape[1]):
                p_x_y.update({
                    (val_x, val_y): (
                    np.sum(X[:, col] == val_x) * p_y[val_y] / np.sum(y == val_y)
                    ) for val_x in np.unique(X[:, col])})
        return p_y, p_x_y

    def _bayes_estimation(self, X, y):
        p_y, p_x_y = dict(), dict()
        K = np.sum(np.unique(y))
        for val_y in np.unique(y):
            p_y[val_y] = (np.sum(y == val_y) + 1) / (y.shape[0] + K)
            for col in range(0, X.shape[1]):
                S_j = np.sum(np.unique(X[:, col]))
                p_x_y.update({
                    (val_x, val_y): (
                    (np.sum(X[:, col] == val_x) * p_y[val_y] + 1) / (np.sum(y == val_y) + S_j)
                    ) for val_x in np.unique(X[:, col])})
        return p_y, p_x_y

    def fit(self, X, y, x):
        if x not in X:
            return print('x is not in dataset!')
        prior_prob, conditional_prob = self._bayes_estimation(X, y)
        posterior_prob = {}
        for val_y in prior_prob.keys():
            product = prior_prob[val_y]
            for val_x in x:
                product *= conditional_prob[(val_x, val_y)]
            posterior_prob[val_y] = product
        prob, class_ = max(zip(posterior_prob.values(), posterior_prob.keys()))
        print((class_, float('%.5f' % prob)))

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
    classify.fit(X, y, np.array([5.9, 2.7]))
