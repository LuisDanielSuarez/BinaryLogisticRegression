import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def _make_predictions(weights, intercept, X):
    return sigmoid(intercept + np.dot(X, weights.T))

def get_cost(weights, intercept, X, y):
    pred_proba = _make_predictions(weights, intercept, X)
    cost = -1/X.shape[0] * sum(y * np.log(pred_proba) + (1 - y) * np.log(1 - pred_proba))
    return np.squeeze(cost)


class BinaryLogReg:

    def __init__(self, alpha=0.01, iterations=700, threshold=0.5, tol=0.0001, max_iter=10000, verbose=False):
        self.alpha = alpha
        self.threshold = threshold
        self.iterations = iterations
        self.max_iter = max_iter
        self.costs = np.array([])
        self.verbose = verbose

    # TODO Add possibility of iterating based on tolerance and max_iter
    # TODO check functionality with pandas Series and DataFrame
    def fit(self, X_train, y_train):
        y_train = y_train.reshape((y_train.shape[0], 1))
        m = X_train.shape[0]
        w = np.zeros(X_train.shape[1]).reshape((1, X_train.shape[1]))
        b = 0
        for i in range(self.iterations):
            z = np.dot(X_train, w.T) + b
            a = sigmoid(z)
            dz = a - y_train
            dw = 1/m * np.dot(dz.T, X_train)
            db = 1/m * np.sum(dz)
            cost = get_cost(w, b, X_train, y_train)
            w = w - self.alpha * dw
            b = b - self.alpha * db

            if i % 50 == 0:
                self.costs = np.append(self.costs, cost)
                if self.verbose:
                    max_iter_digits = int(np.log10(self.max_iter)) + 1
                    print(f'iteration {i:{max_iter_digits}g}, cost: {cost:.4f}')

        self.coefs = w
        self.intercept = b

    def predict_proba(self, test):
        try:
            return _make_predictions(self.coefs, self.intercept, test)
        except AttributeError as atr_err:
            raise RuntimeError("Model wasn't fitted, maybe you want to run .fit() method before?") from atr_err

    def predict(self, test):
        try:
            return int(self.predict_proba(test) > self.threshold)
        except AttributeError as atr_err:
            raise RuntimeError("Model wasn't fitted, maybe you want to run .fit() method before?") from atr_err

    def get_model(self):
        try:
            return {'intercept': self.intercept, 'coefficients': list(self.coefs)}
        except AttributeError as atr_err:
            raise RuntimeError("Model wasn't fitted, maybe you want to run .fit() method before?") from atr_err
