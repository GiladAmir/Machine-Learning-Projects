import numpy as np
import pandas as pd


def pearson_correlation( x, y):
    """
    Calculate the Pearson correlation coefficient for two given columns of data.

    Inputs:
    - x: An array containing a column of m numeric values.
    - y: An array containing a column of m numeric values. 

    Returns:
    - The Pearson correlation coefficient between the two columns.    
    """
    r = 0.0

    mean_y = np.mean(y)
    mean_x = np.mean(x)
    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sqrt(np.sum((x - mean_x) ** 2) * np.sum((y - mean_y) ** 2))

    if denominator == 0:
        return 0

    r = numerator / denominator
    return r


def feature_selection(X, y, n_features=5):
    """
    Select the best features using pearson correlation.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - best_features: list of best features (names - list of strings).  
    """
    best_features = []
    correlations = {}
    for column in X.columns:
        correlations[column] = pearson_correlation(X[column].values, y)

    sorted_features = sorted(correlations, key=lambda k: abs(correlations[k]), reverse=True)
    best_features = sorted_features[:n_features]
    return best_features


class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def sigmoid(self, z):
        """computes the sigmoid function.
        z is the lin combination of the features and their weights as tought in class"""

        return 1 / (1 + np.exp(-z))

    def compute_cost(self, X, y, theta):
        """computes the cost function"""

        m = X.shape[0]
        h = self.sigmoid(X.dot(theta)) #predicted value
        cost = (1/m) * (y.dot(np.log(h))) - (1 - y).dot(np.log(1 - h))
        return cost


    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # set random seed, initialize random theta
        np.random.seed(self.random_state)
        self.theta = np.random.randn(X.shape[1])
        self.thetas.append(self.theta.copy())

        # compute the thetas using gradient descent
        for i in range(self.n_iter):
            h = self.sigmoid(X.dot(self.theta))
            gradient = (1 / len(y)) * np.dot(X.T, h - y)
            self.theta -= self.eta * gradient

            # save theta and cost
            self.thetas.append(self.theta.copy())
            cost = self.compute_cost(X, y, self.theta)
            self.Js.append(cost)

            if i > 0 and abs(self.Js[-2] - self.Js[-1]) < self.eps:
                break



    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        probs = self.sigmoid(X.dot(self.theta))
        preds = np.where(probs >= 0.5, 1, 0)

        return preds

def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = None

    # set random seed
    np.random.seed(random_state)

    # shuffle the data and split into folds
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    fold_size = X.shape[0] // folds
    accuracies = []

    for i in range(folds):
    # create train and test sets
        start = i * fold_size
        end = (i + 1) * fold_size
        X_test = X[start:end]
        y_test = y[start:end]
        X_train = np.concatenate([X[:start], X[end:]], axis=0)
        y_train = np.concatenate([y[:start], y[end:]] , axis=0)

        # initialize the fit model and predict
        algo.fit(X_train, y_train)
        y_pred = algo.predict(X_test)

        # calc accuracy
        accuracy = np.mean(y_pred == y_test)
        accuracies.append(accuracy)

    cv_accuracy = np.mean(accuracies)

    return cv_accuracy


def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = None
    exp = np.exp((-0.5) * (((data - mu) / sigma) ** 2))
    p = (1 / (sigma * np.sqrt(2 * np.pi))) * exp

    return p

class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = []

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """

        n_samples, n_features = data.shape

        # Initialize weights uniformly
        self.weights = np.ones(self.k) / self.k

        # Randomly initialize means using the data
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.mus = data[random_indices]

        # Initialize standard deviations to 1 and the responsibilities to zeros
        self.sigmas = np.ones((self.k, n_features))
        self.responsibilities = np.zeros((n_samples, self.k))



    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        n_samples = data.shape[0]


        for k in range(self.k):
        #loop over each gaussian and calculate the prob with the weight
            #self.responsibilities[:, k] = self.weights[k] * norm_pdf(data, self.mus[k], self.sigmas[k]).flatten()
            pdf = np.prod(norm_pdf(data, self.mus[k], self.sigmas[k]), axis=1)
            self.responsibilities[:, k] = self.weights[k] * pdf

        # normalize resposibilities to sum up to 1
        total_prob = np.sum(self.responsibilities, axis=1).reshape(-1, 1)
        self.responsibilities /= total_prob



    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        n_samples, n_features = data.shape

        # update weights
        effective_n = np.sum(self.responsibilities, axis=0)
        self.weights = effective_n / n_samples

        #update means
        self.mus = np.dot(self.responsibilities.T, data) / effective_n.reshape(-1, 1)

        #update std
        for k in range(self.k):
            diff = data - self.mus[k]
            self.sigmas[k] = np.sqrt(np.sum(self.responsibilities[:, k].reshape(-1, 1) * diff ** 2, axis=0) / effective_n[k])

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        self.init_params(data)
        prev_cost = np.inf

        for i in range(self.n_iter):
            self.expectation(data)
            self.maximization(data)

            log_likelihood = np.sum(np.log(np.sum([self.weights[k] * np.prod(norm_pdf(data, self.mus[k], self.sigmas[k]), axis=1) for k in range(self.k)], axis=0)))
            self.costs.append(-1 * log_likelihood)

            # check for convergence
            if np.abs(self.costs[-1] - prev_cost) < self.eps:
                break
            prev_cost = self.costs[-1]




    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas


def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = np.zeros(data.shape[0])

    for weight, mu, sigma in zip(weights, mus, sigmas):
        pdf += weight * np.prod(norm_pdf(data, mu, sigma), axis=1)

    return pdf

class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = None
        self.gmms = {}
        self.classes = None

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """

        np.random.seed(self.random_state)
        self.classes = np.unique(y)
        self.prior = {cls: np.mean(y == cls) for cls in self.classes}

        for cls in self.classes:
            #only samples that belongs to the current class
            X_cls = X[y == cls]
            gmm = EM(k=self.k, random_state=self.random_state)
            gmm.fit(X_cls)
            self.gmms[cls] = gmm.get_dist_params()


    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        posteriors = []

        for cls in self.classes:
            #weights, mus, sigmas = self.gmms[cls]['weights'], self.gmms[cls]['mus'], self.gmms[cls]['sigmas']
            weights, mus, sigmas = self.gmms[cls]

            likelihood = gmm_pdf(X, weights, mus, sigmas)
            posterior = np.log(self.prior[cls]) + np.log(likelihood)
            posteriors.append(posterior)

        posteriors = np.array(posteriors)

        preds = self.classes[np.argmax(posteriors, axis=0)]

        return preds

def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    from matplotlib import pyplot as plt
    from mlxtend.plotting import plot_decision_regions
    '''
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    '''

    # Logistic Regression
    lor = LogisticRegressionGD(eta=best_eta, eps=best_eps, random_state=1991)
    lor.fit(x_train, y_train)
    lor_train_pred = lor.predict(x_train)
    lor_test_pred = lor.predict(x_test)

    lor_train_acc = np.mean(lor_train_pred == y_train)
    lor_test_acc = np.mean(lor_test_pred == y_test)

    # Plot decision boundary for Logistic Regression
    plt.figure(figsize=(10, 6))
    plot_decision_regions(x_train, y_train, clf=lor)
    plt.title('Logistic Regression Decision Boundary')
    plt.show()

    # Plot cost vs iteration for Logistic Regression
    plt.figure(figsize=(10, 6))
    plt.plot(lor.Js)
    plt.title('Logistic Regression Cost vs Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.show()

    # Naive Bayes
    bayes = NaiveBayesGaussian(k=k, random_state=1991)
    bayes.fit(x_train, y_train)
    bayes_train_pred = bayes.predict(x_train)
    bayes_test_pred = bayes.predict(x_test)

    bayes_train_acc = np.mean(bayes_train_pred == y_train)
    bayes_test_acc = np.mean(bayes_test_pred == y_test)

    # Plot decision boundary for Naive Bayes
    plt.figure(figsize=(10, 6))
    plot_decision_regions(x_train, y_train, clf=bayes)
    plt.title('Naive Bayes Decision Boundary')
    plt.show()


    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}


def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    np.random.seed(1991)

    # Dataset A (Naive Bayes is better)
    mean_a1 = [1, 1, 1]
    cov_a1 = [[1, 0.8, 0.5], [0.8, 1, 0.5], [0.5, 0.5, 1]]
    mean_a2 = [5, 5, 5]
    cov_a2 = [[1, -0.8, -0.5], [-0.8, 1, -0.5], [-0.5, -0.5, 1]]

    data_a1 = multivariate_normal(mean_a1, cov_a1).rvs(500)
    data_a2 = multivariate_normal(mean_a2, cov_a2).rvs(500)

    dataset_a_features = np.vstack((data_a1, data_a2))
    dataset_a_labels = np.hstack((np.zeros(500), np.ones(500)))

    # Dataset B (LR is better)
    mean_b1 = [2, 2, 2]
    cov_b1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    mean_b2 = [4, 4, 4]
    cov_b2 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    data_b1 = multivariate_normal(mean_b1, cov_b1).rvs(500)
    data_b2 = multivariate_normal(mean_b2, cov_b2).rvs(500)

    dataset_b_features = np.vstack((data_b1, data_b2))
    dataset_b_labels = np.hstack((np.zeros(500), np.ones(500)))

    return {
        'dataset_a_features': dataset_a_features,
        'dataset_a_labels': dataset_a_labels,
        'dataset_b_features': dataset_b_features,
        'dataset_b_labels': dataset_b_labels
    }