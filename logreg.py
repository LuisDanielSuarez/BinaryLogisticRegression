import numpy as np


def sigma(z):
    return 1 / (1 + np.exp(-z))


class BinaryLogReg:

    def __init__(self, alpha=0.01, iterations=700, threshold=0.5, tol=0.0001, max_iter=10000):
        self.alpha = alpha
        self.threshold = threshold
        self.iterations = iterations

    # TODO Add possibility of iterating based on tolerance and max_iter
    # TODO check functionality with pandas Series and DataFrame
    def fit(self, X_train, y_train):
        y_train = y_train.reshape((y_train.shape[0], 1))
        m = X_train.shape[0]
        w = np.zeros(X_train.shape[1]).reshape((1, X_train.shape[1]))
        b = 0
        for i in range(self.iterations):
            z = np.dot(X_train, w.T) + b
            a = sigma(z)
            dz = a - y_train
            dw = 1/m * np.dot(dz.T, X_train)
            db = 1/m * np.sum(dz)
            w = w - self.alpha * dw
            b = b - self.alpha * db
        self.coefs = w
        self.intercept = b

    def predict_proba(self, test):
        return sigma(self.intercept + np.dot(test, self.coefs.T))

    def predict(self, test):
        return 1 * (sigma(self.intercept + np.dot(test, self.coefs.T)) > self.threshold)

    def get_model(self):
        return {'intercept': self.intercept, 'coefficients': list(self.coefs)}
