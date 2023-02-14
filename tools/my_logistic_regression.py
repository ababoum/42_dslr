import numpy as np

class MyLogisticRegression():

    """
    Description:
    My personnal logistic regression to classify things.
    """

    def __init__(self, theta, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta
        self.eps = 1e-15

        if not isinstance(self.alpha, (int, float)) or alpha <= 0 or alpha >= 1:
            raise ("Alpha must be strictly between 0 and 1")
        if not isinstance(max_iter, int) or max_iter < 0:
            raise ("Max_iter must be a positive integer")

    @staticmethod
    def is_matrix_valid(x):
        if not isinstance(x, np.ndarray):
            return False
        if len(x.shape) == 1 and x.shape[0] < 1:
            return False
        if len(x.shape) == 2 and (x.shape[0] < 1 or x.shape[1] < 1):
            return False
        if x.size == 0:
            return False
        return True

    @staticmethod
    def is_theta_valid(theta, n):
        if not isinstance(theta, np.ndarray):
            return False
        if len(theta.shape) == 1 and theta.shape != (n,):
            return False
        if len(theta.shape) == 2 and theta.shape != (n, 1):
            return False
        if theta.size == 0:
            return False
        return True

    @staticmethod
    def add_intercept(x):
        """Adds a column of 1's to the non-empty numpy.array x.
        Args:
        x: has to be a numpy.array of dimension m * n.
        Returns:
        X, a numpy.array of dimension m * (n + 1).
        None if x is not a numpy.array.
        None if x is an empty numpy.array.
        Raises:
        This function should not raise any Exception.
        """
        try:
            if not isinstance(x, np.ndarray):
                return None

            new_col = np.empty((x.shape[0], 1))
            np.ndarray.fill(new_col, 1.)

            if len(x.shape) == 1:
                return np.concatenate(
                    [new_col, np.array(list([item] for item in x))], axis=1)
            return np.concatenate([new_col, x], axis=1)
        except:
            return None

    @staticmethod
    def data_spliter(x, y, proportion):
        """Shuffles and splits the dataset (given by x and y) into a training and a test set,
        while respecting the given proportion of examples to be kept in the training set.
        Args:
        x: has to be an numpy.array, a matrix of dimension m * n.
        y: has to be an numpy.array, a vector of dimension m * 1.
        proportion: has to be a float, the proportion of the dataset that will be assigned to the
        training set.
        Return:
        (x_train, x_test, y_train, y_test) as a tuple of numpy.array
        None if x or y is an empty numpy.array.
        None if x and y do not share compatible dimensions.
        None if x, y or proportion is not of expected type.
        Raises:
        This function should not raise any Exception.
        """
        try:
            if not MyLogisticRegression.is_matrix_valid(x) or \
                not MyLogisticRegression.is_matrix_valid(y):
                return None
            if x.shape[0] != y.shape[0]:
                return None
            if not isinstance(proportion, float):
                return None

            # Shuffle the data
            data = np.concatenate([x, y], axis=1)
            np.random.shuffle(data)
            x = data[:, :-1]
            y = data[:, -1].reshape((-1, 1))

            # Split the data
            split = int(x.shape[0] * proportion)
            x_train = x[:split]
            x_test = x[split:]
            y_train = y[:split]
            y_test = y[split:]
            return (x_train, x_test, y_train, y_test)

        except:
            return None

    @staticmethod
    def sigmoid_(x):
        """
        Compute the sigmoid of a vector.
        Args:
        x: has to be a numpy.ndarray of shape (m, 1).
        Returns:
        The sigmoid value as a numpy.ndarray of shape (m, 1).
        None if x is an empty numpy.ndarray.
        Raises:
        This function should not raise any Exception.
        """
        try:
            if not MyLogisticRegression.is_matrix_valid(x) or x.shape[1] != 1:
                return None
            return 1 / (1 + np.exp(-x))
        except:
            return None

    def log_gradient_(self, x, y):
        try:
            if not MyLogisticRegression.is_matrix_valid(x) or \
                not MyLogisticRegression.is_matrix_valid(y) or \
                    not MyLogisticRegression.is_theta_valid(self.theta, x.shape[1] + 1):
                return None
            if x.shape[0] != y.shape[0]:
                return None

            x_p = MyLogisticRegression.add_intercept(x)
            y_pred = self.predict_(x)

            return x_p.T @ (y_pred - y) / x.shape[0]

        except Exception as e:
            print(e)
            return None

    def predict_(self, x):
        if not MyLogisticRegression.is_matrix_valid(x) or \
                not MyLogisticRegression.is_theta_valid(self.theta, x.shape[1] + 1):
            return None

        try:
            x = MyLogisticRegression.add_intercept(x)
            return MyLogisticRegression.sigmoid_(x @ self.theta)
        except:
            return None

    def loss_elem_(self, y, y_hat):
        try:
            if not MyLogisticRegression.is_matrix_valid(y) or \
                    not MyLogisticRegression.is_matrix_valid(y_hat):
                return None
            if y.shape != y_hat.shape or y.shape[1] != 1:
                return None
            y = y.reshape(-1, 1)
            y_hat = y_hat.reshape(-1, 1)

            return - (y * np.log(y_hat + self.eps) + (1 - y) * np.log(1 + self.eps - y_hat))
        except:
            return None

    def loss_(self, y, y_hat):
        try:
            if not MyLogisticRegression.is_matrix_valid(y) or \
                    not MyLogisticRegression.is_matrix_valid(y_hat):
                return None
            y = y.reshape(-1, 1)
            y_hat = y_hat.reshape(-1, 1)
            if y.shape != y_hat.shape or y.shape[1] != 1:
                return None
            elems = self.loss_elem_(y, y_hat)
            if elems is None:
                return None
            return np.sum(elems) / y.shape[0]
        except:
            return None

    def fit_(self, x, y):
        try:
            if not MyLogisticRegression.is_matrix_valid(x) or \
                not MyLogisticRegression.is_matrix_valid(y) or \
                    not MyLogisticRegression.is_theta_valid(self.theta, 1 + x.shape[1]):
                return None

            if x.shape[0] != y.shape[0]:
                return None

            for _ in range(self.max_iter):
                grad = self.log_gradient_(x, y)
                self.theta = self.theta - self.alpha * grad
        except:
            return None
