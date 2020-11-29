import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt


class Perceptron:
    def __init__(self, rate = 0.01, niter = 10):
        self.rate = rate
        self.niter = niter

    def fit(self, X, y):
        """ Fit training data
        X : Training vectors, X.shape : [#samples, #features]
        y : Target values, y.chape: [#samples]"""

        # weights: create a weights array of right size
        # and initialize elements to zero (note: self.weight[0] is the bias)
        self.weight = np.zeros(X.shape[1] + 1)

        # Number of misclassifications, creates an array
        # to hold the number of misclassifications
        self.errors = np.zeros(self.niter)

        # main loop to fit the data to the labels
        for i in range(self.niter):
            # set iteration error to zero
            err = 0

            # loop over all the objects in X and corresponding y element
            for xi, target in zip(X, y):
                # augment current item with 1 for bias
                xiw = np.insert(xi, 0, 1)

                # get predicted output for this item
                output = 1 if np.dot(self.weight, xiw) >= 0 else -1

                # calculate what the current object will add to the weight
                delta_w = self.rate * (target - output) * xiw
                self.weight += delta_w

                # increase the iteration error if delta_w != 0
                if any(delta_w):
                    err += 1

            # Return early if no errors
            if err == 0:
                return self

            # Update the misclassification array with # of errors in iteration
            self.errors[i] = err

        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.weight[1:]) + self.weight[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0, 1, -1)


class LinearRegression:
    def __init__(self):
        self.weight = None

    def fit(self, X, y):
        """
        Fit training data using Least Squares
        X : Training vectors, X.shape : [#samples]
        y : Target values, y.chape: [#samples]"""
        ones = np.ones(np.shape(X)[0])
        Xw = np.c_[ones, X]

        mat1 = np.linalg.inv(np.matmul(Xw.T, Xw))
        mat2 = np.matmul(Xw.T, y)
        self.weight = np.ravel(np.matmul(mat1, mat2))

    def plotData(self, X, y):
        """Plot training data along with my least squares solution"""
        xmin = np.min(X)
        xmax = np.max(X)
        ymin = self.weight[0] + self.weight[1]*xmin
        ymax = self.weight[0] + self.weight[1]*xmax

        plt.plot((xmin, xmax), (ymin, ymax))
        plt.scatter(x=X, y=y)


class PolynomialLinReg:
    def __init__(self):
        self.weight = None

    def fit(self, X, y, dim):
        """
        Fit training data
        X : Training vectors, X.shape : [#samples]
        y : Target values, y.shape: [#samples]
        dim: order of the polynomial (1=linear, 2=quadratic, etc)"""
        ones = np.ones(np.size(X))
        Xmat = np.vstack((ones, X))

        for i in range(2, dim+1):
            Xmat = np.vstack((Xmat, X**i))

        mat1 = np.linalg.inv(np.matmul(Xmat, Xmat.T))
        mat2 = np.matmul(Xmat, y.T)
        self.weight = np.matmul(mat1, mat2)


class DecisionStump:
    def ERM(self, X, y, D):
        """
        Performs ERM for decision stumps. Returns the dimension to check and
        the value to check against
        X : Training vectors, X.shape : [#dimensions, #samples]
        y : Target values, y.chape: [#samples]
        D : Distribution vector (how likely each X value is to appear), D.shape: [#samples]
        """
        # Get list of points and initialize loop variables
        points = [(X[:, i], y[i]) for i in range(len(y))]
        F_min = -1
        j_min = -1
        theta = []

        for j in range(X.shape[0]):
            # Sort by j'th coordinate and append extra X value
            comp = lambda p: p[0][j]
            points = sorted(points, key=comp)
            xx = list(X[j])
            xx.append(xx[-1]+1)

            # Calculate F by summing D values where y > 0
            yD = np.multiply(y, D)
            F = np.sum(yD[yD>0])

            # Check if this is a new best stump point
            if F < F_min or F_min == -1:
                F_min = F
                theta = xx[0]-1
                j_min = j

            # Walk through points, seeing if any gives a better stump point
            for i in range(y.size):
                F -= yD[i]
                if F < F_min and xx[i] != xx[i+1]:
                    F_min = F
                    theta = (xx[i] + xx[i+1])/2
                    j_min = j

        return j_min, theta


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
               np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(x=X[y == c1, 0], y=X[y == c1, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=c1)
