from ML import *
from unittest import TestCase
import pandas as pd, numpy as np

class Test_ML(TestCase):
    def test_Perceptron(self):
        pn = Perceptron(0.1, 10)

        # Test with three variables
        df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-'
                         'databases/iris/iris.data', header=None)

        X = df.iloc[0:100, [1, 2, 3]].values
        y = df.iloc[0:100, 4].values
        y = np.where(y == 'Iris-setosa', -1, 1)
        pn.fit(X, y)
        self.assertTrue(np.allclose(pn.weight, [-0.2, -0.82, 1.04, 0.44]))
        self.assertTrue(np.allclose(pn.errors, [2, 2, 1, 0, 0, 0, 0, 0, 0, 0]))

        # Test with two variables
        X = df.iloc[0:100, [0, 2]].values
        y = df.iloc[0:100, 4].values
        y = np.where(y == 'Iris-setosa', -1, 1)
        pn.fit(X, y)
        self.assertTrue(np.allclose(pn.weight, [-0.4, -0.68, 1.82]))
        self.assertTrue(np.allclose(pn.errors, [2, 2, 3, 2, 1, 0, 0, 0, 0, 0]))

        # Plot it
        plot_decision_regions(X, y, pn)
        plt.xlabel("sepal length [cm]")
        plt.ylabel("petal length [cm]")
        plt.show()

    def test_LinearRegression(self):
        lr = LinearRegression()

        # Test with simple data: y = 4 + 2x
        X = np.array([0, 1, 2, 3, 4])
        y = np.array([4, 6, 8, 10, 12])
        lr.fit(X, y)
        self.assertTrue(np.allclose(lr.weight, [4, 2]))

        # Test with iris data
        df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-'
                         'databases/iris/iris.data', header=None)

        X = df.iloc[50:100, [0]].values
        y = df.iloc[50:100, [2]].values
        lr.fit(X, y)
        self.assertTrue(np.allclose(lr.weight, [0.18511551, 0.68646976]))

        # Plot it
        lr.plotData(X, y)
        plt.show()


    def test_DecisionStumps(self):
        ds = DecisionStump()

        # Test with simple data
        # Notice only the second coordinate changes, and y = -1 when X[1] > 2.5
        X = np.array([[5, 5, 5, 5, 5], [0, 1, 2, 3, 4], [1, 1, 1, 1, 1]])
        y = np.array([1, 1, 1, -1, -1])

        resp = ds.ERM(X, y, [0.2]*5)
        self.assertEqual(resp, (1, 2.5))