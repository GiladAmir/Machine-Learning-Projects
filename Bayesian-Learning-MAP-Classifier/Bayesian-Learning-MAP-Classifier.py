import numpy as np


class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        self.X_Y = {
            (0, 0): 0.2,
            (0, 1): 0.1,
            (1, 0): 0.1,
            (1, 1): 0.6

        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0): 0.1,
            (0, 1): 0.1,
            (1, 0): 0.4,
            (1, 1): 0.4
        }  # P(X=x, C=y)

        self.Y_C = {
            (0, 0): 0.2,
            (0, 1): 0.2,
            (1, 0): 0.3,
            (1, 1): 0.3
        }  # P(Y=y, C=c)

        self.X_Y_C = {
            (0, 0, 0): 0.05,
            (0, 0, 1): 0.05,
            (0, 1, 0): 0.14,
            (0, 1, 1): 0.14,
            (1, 0, 0): 0.14,
            (1, 0, 1): 0.14,
            (1, 1, 0): 0.17,
            (1, 1, 1): 0.17,
        }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y

        # Check for every key of X_Y if X and Y are dependent
        for x, y in X_Y.keys():
            if np.isclose(X[x] * Y[y], X_Y[(x, y)]):
                return False

        # Exist the loop after check each x for X and y for Y
        return True

    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndendent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C

        # Check for every key of X_Y_C if X_C and Y_C are dependent
        for x, y, c in X_Y_C.keys():
            if not np.isclose(X_C[(x, c)] * Y_C[(y, c)], X_Y_C[(x, y, c)]):
                return False

        # Exist the loop after check each x&c for X_C and y&c for Y_C
        return True


def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """
    log_p = None
    # Calculate the poisson distribution for X=k and lambda=rate
    poisson = np.power(rate, k) * np.exp(-rate) / factorial(k)
    # Calculate the log of the poisson distribution
    log_p = np.log(poisson)

    return log_p


def factorial(n):
    """
    Helper function.
    Args:
        n: The integer we calculate the factorial

    Returns: The factorial of n

    """

    # Calculate the factorial of n
    res = 1

    for i in range(2, n + 1):
        res *= i
    return res


def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """
    likelihoods = []

    # Check the likelihood for each sample and rate
    for rate in rates:
        log_likelyhood = 0
        # Sum the log likelihoods of each sample
        for sample in samples:
            log_likelyhood += poisson_log_pmf(sample, rate)
        # Add the total log likelihoods of each sample
        likelihoods.append(log_likelyhood)

    return likelihoods


def possion_iterative_mle(samples, rates):
    """
    samples: set of univariate discrete observations
    rate: a rate to calculate log-likelihood by.

    return: the rate that maximizes the likelihood
    """
    rate = 0.0
    likelihoods = get_poisson_log_likelihoods(samples, rates)  # might help

    # Find the  maximum likelihoods which was yielded from the best rate
    max_rate = max(likelihoods)
    # Find the index of the best rate
    max_index = likelihoods.index(max_rate)
    # Get the rate corresponding to the maximum likelihood
    rate = rates[max_index]

    return rate


def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    mean = None

    # In poisson distribution, the rate parameter is equal to the mean
    mean = np.mean(samples)

    return mean


def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.

    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.

    Returns the normal distribution pdf according to the given mean and std for the given x.
    """
    p = None
    # Calculate the squre root part
    squre_root = np.sqrt(2 * np.pi * std)
    # Calculate the exponent part
    e_pow = np.exp(-1 * ((x - mean) ** 2) / (2 * std ** 2))
    # Calculate the distribution
    p = e_pow / squre_root
    return p


class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.

        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class to calculate the parameters for.
        """
        # Extract the data for the specific class
        self.data = dataset
        self.class_value = class_value
        class_data = dataset[dataset[:, -1] == class_value][:, :-1]
        self.class_data = class_data

        # Calculate the mean and standard deviation for each feature
        self.mean = np.mean(class_data, axis=0)
        self.std = np.std(class_data, axis=0)

    def get_prior(self):
        """
        Returns the prior probability of the class according to the dataset distribution.
        """
        prior = None

        # Number of samples for the specific class
        class_samples_count = len(self.class_data)
        # Total number of samples in the dataset
        total_samples_count = len(self.data)
        # Calculate the prior probability
        prior = class_samples_count / total_samples_count

        return prior

    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        likelihood = None
        likelihoods = (1 / (np.sqrt(2 * np.pi) * self.std)) * np.exp(-0.5 * ((x - self.mean) ** 2) / (self.std ** 2))
        likelihood = np.prod(likelihoods)

        return likelihood

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None

        prior = self.get_prior()
        likelihood = self.get_instance_likelihood(x)
        # The posterior is prior*likelihood
        posterior = likelihood * prior
        return posterior


class MAPClassifier():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum a posteriori classifier.
        This class will hold 2 class distributions.
        One for class 0 and one for class 1, and will predict an instance
        using the class that outputs the highest posterior probability
        for the given instance.

        Input
            - ccd0 : An object contating the relevant parameters and methods
                     for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods
                     for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.

        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        # Get the posterior probabilities for both classes
        posterior0 = self.ccd0.get_instance_posterior(x)
        posterior1 = self.ccd1.get_instance_posterior(x)

        # Compare the posterior probabilities and predict the class
        if posterior0 > posterior1:
            pred = 0
        else:
            pred = 1

        return pred


def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given a test_set using a MAP classifier object.

    Input
        - test_set: The test_set for which to compute the accuracy (Numpy array). where the class label is the last column
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.

    Ouput
        - Accuracy = #Correctly Classified / test_set size
    """
    acc = None
    correct_predictions = 0
    total_predictions = len(test_set)

    for instance in test_set:
        # The feature vector is all columns except the last one
        features = instance[:-1]
        # The actual class label is the last column
        actual_class = instance[-1]

        # Predict the class using the MAPClassifier
        predicted_class = map_classifier.predict(features)

        # Compare the predicted class with the actual class
        if predicted_class == actual_class:
            correct_predictions += 1

    # Calculate accuracy
    acc = correct_predictions / total_predictions
    return acc


def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variable normal desnity function for a given x, mean and covarince matrix.

    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.

    Returns the normal distribution pdf according to the given mean and var for the given x.
    """
    pdf = None
    k = len(mean)  # Dimensionality of the data
    x = np.array(x)
    mean = np.array(mean)
    cov = np.array(cov)

    # Compute the normalization term
    norm_term = 1 / (np.sqrt((2 * np.pi) ** k * np.linalg.det(cov)))

    # Compute the exponent term
    x_minus_mean = x - mean
    inv_cov = np.linalg.inv(cov)
    exponent_term = np.exp(-0.5 * np.dot(np.dot(x_minus_mean.T, inv_cov), x_minus_mean))

    # Combine the terms to get the PDF
    pdf = norm_term * exponent_term
    return pdf


class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditional multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.

        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """

        # Extract the data for the specified class
        self.data = dataset
        class_data = dataset[dataset[:, -1] == class_value][:, :-1]
        self.class_data = class_data

        # Calculate the mean vector for the given class
        self.mean = np.mean(class_data, axis=0)

        # Calculate the covariance matrix for the given class
        self.cov = np.cov(class_data.T)

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = None

        # Number of samples for the specific class
        class_samples = self.class_data.shape[0]
        total_samples = self.data.shape[0]

        # Calculate the prior probability
        prior = class_samples / total_samples

        return prior

    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset distribution.
        """
        likelihood = None
        likelihood = multi_normal_pdf(x, self.mean, self.cov)
        return likelihood

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None

        prior = self.get_prior()
        likelihood = self.get_instance_likelihood(x)

        # The posterior is prior*likelihood
        posterior = likelihood * prior

        return posterior


class MaxPrior():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum prior classifier.
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest prior probability for the given instance.

        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.

        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None

        prior0 = self.ccd0.get_prior()
        prior1 = self.ccd1.get_prior()

        # Compare the priors probabilities and predict the class
        if prior0 > prior1:
            pred = 0
        else:
            pred = 1

        return pred


class MaxLikelihood():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum Likelihood classifier.
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest likelihood probability for the given instance.

        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.

        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None

        likelihood0 = self.ccd0.get_instance_likelihood(x)
        likelihood1 = self.ccd1.get_instance_likelihood(x)

        # Compare the likelihoods probabilities and predict the class
        if likelihood0 > likelihood1:
            pred = 0
        else:
            pred = 1

        return pred


EPSILLON = 1e-6  # if a certain value only occurs in the test set, the probability for that value will be EPSILLON.


class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes
        distribution for a specific class. The probabilites are computed with laplace smoothing.

        Input
        - dataset: The dataset as a numpy array.
        - class_value: Compute the relevant parameters only for instances from the given class.
        """
        # Extract the data for the specific class
        self.data = dataset
        class_data = dataset[dataset[:, -1] == class_value][:, :-1]
        self.class_data = class_data
        self.class_value = class_value
        # Get the number of features
        self.num_features = self.class_data.shape[1]
        # Filed of a helper function
        self.cond_probs = self.compute_conditional_probabilities()

    def get_prior(self):
        """
        Returns the prior porbability of the class
        according to the dataset distribution.
        """
        prior = None

        # Number of samples for the specific class
        class_samples_count = len(self.class_data)
        # Total number of samples in the dataset
        total_samples_count = len(self.data)
        # Calculate the prior probability
        prior = class_samples_count / total_samples_count

        return prior

    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under
        the class according to the dataset distribution.
        """
        likelihood = None
        # Initialize likelihood to be 1 for the multiplication
        likelihood = 1.0
        # Iterate over each feature
        for i in range(self.num_features):
            feature_val = x[i]
            feature_probs = self.cond_probs[i]
            likelihood *= feature_probs.get(feature_val, 1 / (len(self.class_data) + len(feature_probs)))
        return likelihood

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance
        under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None

        prior = self.get_prior()
        likelihood = self.get_instance_likelihood(x)
        # The posterior is prior*likelihood
        posterior = likelihood * prior

        return posterior

    def compute_conditional_probabilities(self):
        """
        Helper function.
        Computes conditional probabilities.
        """
        cond_probs = []  # List to store the conditional probabilities for each feature
        # Iterate over each feature in the data
        for i in range(self.num_features):
            feature_vals = self.class_data[:, i]
            unique_vals, counts = np.unique(feature_vals, return_counts=True)
            feature_prob = {}
            total_counts = len(feature_vals)

            # Iterate over each unique value and its count
            for val, count in zip(unique_vals, counts):
                # Compute the conditional probability
                feature_prob[val] = (count + 1) / (total_counts + len(unique_vals))
            cond_probs.append(feature_prob)
        return cond_probs


class MAPClassifier_DNB():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum a posteriori classifier.
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest posterior probability for the given instance.

        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.

        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None

        posterior0 = self.ccd0.get_instance_posterior(x)
        posterior1 = self.ccd1.get_instance_posterior(x)

        # Compare the posterior probabilities and predict the class
        if posterior0 > posterior1:
            pred = 0
        else:
            pred = 1
        return pred

    def compute_accuracy(self, test_set):
        """
        Compute the accuracy of a given a testset using a MAP classifier object.

        Input
            - test_set: The test_set for which to compute the accuracy (Numpy array).
        Ouput
            - Accuracy = #Correctly Classified / #test_set size
        """
        acc = None
        correct_predictions = 0

        for instance in test_set:
            # The feature vector is all columns except the last one
            features = instance[:-1]
            # The actual class label is the last column
            actual_class = instance[-1]

            # Predict the class using the MAPClassifier
            predicted_class = self.predict(features)

            # Compare the predicted class with the actual class
            if predicted_class == actual_class:
                correct_predictions += 1

        # Calculate accuracy
        acc = correct_predictions / test_set.shape[0]
        return acc


import numpy as np


class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        self.X_Y = {
            (0, 0): 0.2,
            (0, 1): 0.1,
            (1, 0): 0.1,
            (1, 1): 0.6

        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0): 0.1,
            (0, 1): 0.1,
            (1, 0): 0.4,
            (1, 1): 0.4
        }  # P(X=x, C=y)

        self.Y_C = {
            (0, 0): 0.2,
            (0, 1): 0.2,
            (1, 0): 0.3,
            (1, 1): 0.3
        }  # P(Y=y, C=c)

        self.X_Y_C = {
            (0, 0, 0): 0.05,
            (0, 0, 1): 0.05,
            (0, 1, 0): 0.14,
            (0, 1, 1): 0.14,
            (1, 0, 0): 0.14,
            (1, 0, 1): 0.14,
            (1, 1, 0): 0.17,
            (1, 1, 1): 0.17,
        }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y

        # Check for every key of X_Y if X and Y are dependent
        for x, y in X_Y.keys():
            if np.isclose(X[x] * Y[y], X_Y[(x, y)]):
                return False

        # Exist the loop after check each x for X and y for Y
        return True

    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndendent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C

        # Check for every key of X_Y_C if X_C and Y_C are dependent
        for x, y, c in X_Y_C.keys():
            if not np.isclose(X_C[(x, c)] * Y_C[(y, c)], X_Y_C[(x, y, c)]):
                return False

        # Exist the loop after check each x&c for X_C and y&c for Y_C
        return True


def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """
    log_p = None
    # Calculate the poisson distribution for X=k and lambda=rate
    poisson = np.power(rate, k) * np.exp(-rate) / factorial(k)
    # Calculate the log of the poisson distribution
    log_p = np.log(poisson)

    return log_p


def factorial(n):
    """
    Helper function.
    Args:
        n: The integer we calculate the factorial

    Returns: The factorial of n

    """

    # Calculate the factorial of n
    res = 1

    for i in range(2, n + 1):
        res *= i
    return res


def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """
    likelihoods = []

    # Check the likelihood for each sample and rate
    for rate in rates:
        log_likelyhood = 0
        # Sum the log likelihoods of each sample
        for sample in samples:
            log_likelyhood += poisson_log_pmf(sample, rate)
        # Add the total log likelihoods of each sample
        likelihoods.append(log_likelyhood)

    return likelihoods


def possion_iterative_mle(samples, rates):
    """
    samples: set of univariate discrete observations
    rate: a rate to calculate log-likelihood by.

    return: the rate that maximizes the likelihood
    """
    rate = 0.0
    likelihoods = get_poisson_log_likelihoods(samples, rates)  # might help

    # Find the  maximum likelihoods which was yielded from the best rate
    max_rate = max(likelihoods)
    # Find the index of the best rate
    max_index = likelihoods.index(max_rate)
    # Get the rate corresponding to the maximum likelihood
    rate = rates[max_index]

    return rate


def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    mean = None

    # In poisson distribution, the rate parameter is equal to the mean
    mean = np.mean(samples)

    return mean


def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.

    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.

    Returns the normal distribution pdf according to the given mean and std for the given x.
    """
    p = None
    # Calculate the squre root part
    squre_root = np.sqrt(2 * np.pi * std)
    # Calculate the exponent part
    e_pow = np.exp(-1 * ((x - mean) ** 2) / (2 * std ** 2))
    # Calculate the distribution
    p = e_pow / squre_root
    return p


class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.

        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class to calculate the parameters for.
        """
        # Extract the data for the specific class
        self.data = dataset
        self.class_value = class_value
        class_data = dataset[dataset[:, -1] == class_value][:, :-1]
        self.class_data = class_data

        # Calculate the mean and standard deviation for each feature
        self.mean = np.mean(class_data, axis=0)
        self.std = np.std(class_data, axis=0)

    def get_prior(self):
        """
        Returns the prior probability of the class according to the dataset distribution.
        """
        prior = None

        # Number of samples for the specific class
        class_samples_count = len(self.class_data)
        # Total number of samples in the dataset
        total_samples_count = len(self.data)
        # Calculate the prior probability
        prior = class_samples_count / total_samples_count

        return prior

    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        likelihood = None
        likelihoods = (1 / (np.sqrt(2 * np.pi) * self.std)) * np.exp(-0.5 * ((x - self.mean) ** 2) / (self.std ** 2))
        likelihood = np.prod(likelihoods)

        return likelihood

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None

        prior = self.get_prior()
        likelihood = self.get_instance_likelihood(x)
        # The posterior is prior*likelihood
        posterior = likelihood * prior
        return posterior


class MAPClassifier():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum a posteriori classifier.
        This class will hold 2 class distributions.
        One for class 0 and one for class 1, and will predict an instance
        using the class that outputs the highest posterior probability
        for the given instance.

        Input
            - ccd0 : An object contating the relevant parameters and methods
                     for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods
                     for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.

        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        # Get the posterior probabilities for both classes
        posterior0 = self.ccd0.get_instance_posterior(x)
        posterior1 = self.ccd1.get_instance_posterior(x)

        # Compare the posterior probabilities and predict the class
        if posterior0 > posterior1:
            pred = 0
        else:
            pred = 1

        return pred


def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given a test_set using a MAP classifier object.

    Input
        - test_set: The test_set for which to compute the accuracy (Numpy array). where the class label is the last column
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.

    Ouput
        - Accuracy = #Correctly Classified / test_set size
    """
    acc = None
    correct_predictions = 0
    total_predictions = len(test_set)

    for instance in test_set:
        # The feature vector is all columns except the last one
        features = instance[:-1]
        # The actual class label is the last column
        actual_class = instance[-1]

        # Predict the class using the MAPClassifier
        predicted_class = map_classifier.predict(features)

        # Compare the predicted class with the actual class
        if predicted_class == actual_class:
            correct_predictions += 1

    # Calculate accuracy
    acc = correct_predictions / total_predictions
    return acc


def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variable normal desnity function for a given x, mean and covarince matrix.

    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.

    Returns the normal distribution pdf according to the given mean and var for the given x.
    """
    pdf = None
    k = len(mean)  # Dimensionality of the data
    x = np.array(x)
    mean = np.array(mean)
    cov = np.array(cov)

    # Compute the normalization term
    norm_term = 1 / (np.sqrt((2 * np.pi) ** k * np.linalg.det(cov)))

    # Compute the exponent term
    x_minus_mean = x - mean
    inv_cov = np.linalg.inv(cov)
    exponent_term = np.exp(-0.5 * np.dot(np.dot(x_minus_mean.T, inv_cov), x_minus_mean))

    # Combine the terms to get the PDF
    pdf = norm_term * exponent_term
    return pdf


class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditional multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.

        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """

        # Extract the data for the specified class
        self.data = dataset
        class_data = dataset[dataset[:, -1] == class_value][:, :-1]
        self.class_data = class_data

        # Calculate the mean vector for the given class
        self.mean = np.mean(class_data, axis=0)

        # Calculate the covariance matrix for the given class
        self.cov = np.cov(class_data.T)

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = None

        # Number of samples for the specific class
        class_samples = self.class_data.shape[0]
        total_samples = self.data.shape[0]

        # Calculate the prior probability
        prior = class_samples / total_samples

        return prior

    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset distribution.
        """
        likelihood = None
        likelihood = multi_normal_pdf(x, self.mean, self.cov)
        return likelihood

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None

        prior = self.get_prior()
        likelihood = self.get_instance_likelihood(x)

        # The posterior is prior*likelihood
        posterior = likelihood * prior

        return posterior


class MaxPrior():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum prior classifier.
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest prior probability for the given instance.

        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.

        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None

        prior0 = self.ccd0.get_prior()
        prior1 = self.ccd1.get_prior()

        # Compare the priors probabilities and predict the class
        if prior0 > prior1:
            pred = 0
        else:
            pred = 1

        return pred


class MaxLikelihood():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum Likelihood classifier.
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest likelihood probability for the given instance.

        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.

        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None

        likelihood0 = self.ccd0.get_instance_likelihood(x)
        likelihood1 = self.ccd1.get_instance_likelihood(x)

        # Compare the likelihoods probabilities and predict the class
        if likelihood0 > likelihood1:
            pred = 0
        else:
            pred = 1

        return pred


EPSILLON = 1e-6  # if a certain value only occurs in the test set, the probability for that value will be EPSILLON.


class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes
        distribution for a specific class. The probabilites are computed with laplace smoothing.

        Input
        - dataset: The dataset as a numpy array.
        - class_value: Compute the relevant parameters only for instances from the given class.
        """
        # Extract the data for the specific class
        self.data = dataset
        class_data = dataset[dataset[:, -1] == class_value][:, :-1]
        self.class_data = class_data
        self.class_value = class_value
        # Get the number of features
        self.num_features = self.class_data.shape[1]
        # Filed of a helper function
        self.cond_probs = self.compute_conditional_probabilities()

    def get_prior(self):
        """
        Returns the prior porbability of the class
        according to the dataset distribution.
        """
        prior = None

        # Number of samples for the specific class
        class_samples_count = len(self.class_data)
        # Total number of samples in the dataset
        total_samples_count = len(self.data)
        # Calculate the prior probability
        prior = class_samples_count / total_samples_count

        return prior

    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under
        the class according to the dataset distribution.
        """
        likelihood = None
        # Initialize likelihood to be 1 for the multiplication
        likelihood = 1.0
        # Iterate over each feature
        for i in range(self.num_features):
            feature_val = x[i]
            feature_probs = self.cond_probs[i]
            likelihood *= feature_probs.get(feature_val, 1 / (len(self.class_data) + len(feature_probs)))
        return likelihood

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance
        under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None

        prior = self.get_prior()
        likelihood = self.get_instance_likelihood(x)
        # The posterior is prior*likelihood
        posterior = likelihood * prior

        return posterior

    def compute_conditional_probabilities(self):
        """
        Helper function.
        Computes conditional probabilities.
        """
        cond_probs = []  # List to store the conditional probabilities for each feature
        # Iterate over each feature in the data
        for i in range(self.num_features):
            feature_vals = self.class_data[:, i]
            unique_vals, counts = np.unique(feature_vals, return_counts=True)
            feature_prob = {}
            total_counts = len(feature_vals)

            # Iterate over each unique value and its count
            for val, count in zip(unique_vals, counts):
                # Compute the conditional probability
                feature_prob[val] = (count + 1) / (total_counts + len(unique_vals))
            cond_probs.append(feature_prob)
        return cond_probs


class MAPClassifier_DNB():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum a posteriori classifier.
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest posterior probability for the given instance.

        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.

        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None

        posterior0 = self.ccd0.get_instance_posterior(x)
        posterior1 = self.ccd1.get_instance_posterior(x)

        # Compare the posterior probabilities and predict the class
        if posterior0 > posterior1:
            pred = 0
        else:
            pred = 1
        return pred

    def compute_accuracy(self, test_set):
        """
        Compute the accuracy of a given a testset using a MAP classifier object.

        Input
            - test_set: The test_set for which to compute the accuracy (Numpy array).
        Ouput
            - Accuracy = #Correctly Classified / #test_set size
        """
        acc = None
        correct_predictions = 0

        for instance in test_set:
            # The feature vector is all columns except the last one
            features = instance[:-1]
            # The actual class label is the last column
            actual_class = instance[-1]

            # Predict the class using the MAPClassifier
            predicted_class = self.predict(features)

            # Compare the predicted class with the actual class
            if predicted_class == actual_class:
                correct_predictions += 1

        # Calculate accuracy
        acc = correct_predictions / test_set.shape[0]
        return acc

# import numpy as np
# import math
#
# class conditional_independence():
#
#     def __init__(self):
#
#         # You need to fill the None value with *valid* probabilities
#         self.X = {0: 0.3, 1: 0.7}  # P(X=x)
#         self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
#         self.C = {0: 0.5, 1: 0.5}  # P(C=c)
#
#         self.X_Y = {
#             (0, 0): 0.09,
#             (0, 1): 0.21,
#             (1, 0): 0.21,
#             (1, 1): 0.49
#         }  # P(X=x, Y=y)
#
#         self.X_C = {
#             (0, 0): 0.15,
#             (0, 1): 0.15,
#             (1, 0): 0.35,
#             (1, 1): 0.35
#         }  # P(X=x, C=y)
#
#         self.Y_C = {
#             (0, 0): 0.15,
#             (0, 1): 0.15,
#             (1, 0): 0.35,
#             (1, 1): 0.35
#         }  # P(Y=y, C=c)
#
#         self.X_Y_C = {
#             (0, 0, 0): 0.045,
#             (0, 0, 1): 0.045,
#             (0, 1, 0): 0.105,
#             (0, 1, 1): 0.105,
#             (1, 0, 0): 0.105,
#             (1, 0, 1): 0.105,
#             (1, 1, 0): 0.245,
#             (1, 1, 1): 0.245,
#         }  # P(X=x, Y=y, C=c)
#
#     def is_X_Y_dependent(self):
#         """
#         return True iff X and Y are depndendent
#         """
#         X = self.X
#         Y = self.Y
#         X_Y = self.X_Y
#
#         # Check for every key of X_Y if X and Y are dependent
#         for x, y in X_Y.keys():
#             if np.isclose(X[x] * Y[y], X_Y[(x, y)]):
#                 return False
#
#         # Exist the loop after check each x for X and y for Y
#         return True
#
#     def is_X_Y_given_C_independent(self):
#         """
#         return True iff X_given_C and Y_given_C are indepndendent
#         """
#         X = self.X
#         Y = self.Y
#         C = self.C
#         X_C = self.X_C
#         Y_C = self.Y_C
#         X_Y_C = self.X_Y_C
#
#         # Check for every key of X_Y_C if X_C and Y_C are dependent
#         for x, y, c in X_Y_C.keys():
#             if not np.isclose(X_C[(x, c)] * Y_C[(y, c)], X_Y_C[(x, y, c)]):
#                 return False
#
#         # Exist the loop after check each x&c for X_C and y&c for Y_C
#         return True
#
#
# def poisson_log_pmf(k, rate):
#     """
#     k: A discrete instance
#     rate: poisson rate parameter (lambda)
#
#     return the log pmf value for instance k given the rate
#     """
#     log_p = None
#     # Calculate the poisson distribution for X=k and lambda=rate
#     poisson = np.power(rate, k) * np.exp(-rate) / factorial(k)
#     # Calculate the log of the poisson distribution
#     log_p = np.log(poisson)
#
#     return log_p
#
#
# def factorial(n):
#     """
#     Helper function.
#     Args:
#         n: The integer we calculate the factorial
#
#     Returns: The factorial of n
#
#     """
#
#     # Calculate the factorial of n
#     res = 1
#
#     for i in range(2, n + 1):
#         res *= i
#     return res
#
#
# def get_poisson_log_likelihoods(samples, rates):
#     """
#     samples: set of univariate discrete observations
#     rates: an iterable of rates to calculate log-likelihood by.
#
#     return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
#     """
#     likelihoods = []
#
#     # Check the likelihood for each sample and rate
#     for rate in rates:
#         log_likelyhood = 0
#         # Sum the log likelihoods of each sample
#         for sample in samples:
#             log_likelyhood += poisson_log_pmf(sample, rate)
#         # Add the total log likelihoods of each sample
#         likelihoods.append(log_likelyhood)
#
#     return likelihoods
#
#
# def possion_iterative_mle(samples, rates):
#     """
#     samples: set of univariate discrete observations
#     rate: a rate to calculate log-likelihood by.
#
#     return: the rate that maximizes the likelihood
#     """
#     rate = 0.0
#     likelihoods = get_poisson_log_likelihoods(samples, rates)  # might help
#
#     # Find the  maximum likelihoods which was yielded from the best rate
#     max_rate = max(likelihoods)
#     # Find the index of the best rate
#     max_index = likelihoods.index(max_rate)
#     # Get the rate corresponding to the maximum likelihood
#     rate = rates[max_index]
#
#     return rate
#
#
# def possion_analytic_mle(samples):
#     """
#     samples: set of univariate discrete observations
#
#     return: the rate that maximizes the likelihood
#     """
#     mean = None
#
#     # In poisson distribution, the rate parameter is equal to the mean
#     mean = np.mean(samples)
#
#     return mean
#
#
# def normal_pdf(x, mean, std):
#     """
#     Calculate normal desnity function for a given x, mean and standrad deviation.
#
#     Input:
#     - x: A value we want to compute the distribution for.
#     - mean: The mean value of the distribution.
#     - std:  The standard deviation of the distribution.
#
#     Returns the normal distribution pdf according to the given mean and std for the given x.
#     """
#     p = None
#     # Calculate the squre root part
#     squre_root = np.sqrt(2 * np.pi * std)
#     # Calculate the exponent part
#     e_pow = np.exp(-1 * ((x - mean) ** 2) / (2 * std ** 2))
#     # Calculate the distribution
#     p = e_pow / squre_root
#     return p
#
#
# class NaiveNormalClassDistribution():
#     def __init__(self, dataset, class_value):
#         """
#         A class which encapsulates the relevant parameters(mean, std) for a class conditinoal normal distribution.
#         The mean and std are computed from a given data set.
#
#         Input
#         - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
#         - class_value : The class to calculate the parameters for.
#         """
#         # Extract the data for the specific class
#         self.data = dataset
#         self.class_value = class_value
#         class_data = dataset[dataset[:, -1] == class_value][:, :-1]
#         self.class_data = class_data
#
#         # Calculate the mean and standard deviation for each feature
#         self.mean = np.mean(class_data, axis=0)
#         self.std = np.std(class_data, axis=0)
#
#     def get_prior(self):
#         """
#         Returns the prior probability of the class according to the dataset distribution.
#         """
#         prior = None
#
#         # Number of samples for the specific class
#         class_samples_count = len(self.class_data)
#         # Total number of samples in the dataset
#         total_samples_count = len(self.data)
#         # Calculate the prior probability
#         prior = class_samples_count / total_samples_count
#
#         return prior
#
#     def get_instance_likelihood(self, x):
#         """
#         Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
#         """
#         likelihood = None
#         likelihoods = (1 / (np.sqrt(2 * np.pi) * self.std)) * np.exp(-0.5 * ((x - self.mean) ** 2) / (self.std ** 2))
#         likelihood = np.prod(likelihoods)
#
#         return likelihood
#
#     def get_instance_posterior(self, x):
#         """
#         Returns the posterior porbability of the instance under the class according to the dataset distribution.
#         * Ignoring p(x)
#         """
#         posterior = None
#
#         prior = self.get_prior()
#         likelihood = self.get_instance_likelihood(x)
#         # The posterior is prior*likelihood
#         posterior = likelihood * prior
#         return posterior
#
#
# class MAPClassifier():
#     def __init__(self, ccd0, ccd1):
#         """
#         A Maximum a posteriori classifier.
#         This class will hold 2 class distributions.
#         One for class 0 and one for class 1, and will predict an instance
#         using the class that outputs the highest posterior probability
#         for the given instance.
#
#         Input
#             - ccd0 : An object contating the relevant parameters and methods
#                      for the distribution of class 0.
#             - ccd1 : An object contating the relevant parameters and methods
#                      for the distribution of class 1.
#         """
#         self.ccd0 = ccd0
#         self.ccd1 = ccd1
#
#     def predict(self, x):
#         """
#         Predicts the instance class using the 2 distribution objects given in the object constructor.
#
#         Input
#             - An instance to predict.
#         Output
#             - 0 if the posterior probability of class 0 is higher and 1 otherwise.
#         """
#         pred = None
#         # Get the posterior probabilities for both classes
#         posterior0 = self.ccd0.get_instance_posterior(x)
#         posterior1 = self.ccd1.get_instance_posterior(x)
#
#         # Compare the posterior probabilities and predict the class
#         if posterior0 > posterior1:
#             pred = 0
#         else:
#             pred = 1
#
#         return pred
#
#
# def compute_accuracy(test_set, map_classifier):
#     """
#     Compute the accuracy of a given a test_set using a MAP classifier object.
#
#     Input
#         - test_set: The test_set for which to compute the accuracy (Numpy array). where the class label is the last column
#         - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
#
#     Ouput
#         - Accuracy = #Correctly Classified / test_set size
#     """
#     acc = None
#     correct_predictions = 0
#     total_predictions = len(test_set)
#
#     for instance in test_set:
#         # The feature vector is all columns except the last one
#         features = instance[:-1]
#         # The actual class label is the last column
#         actual_class = instance[-1]
#
#         # Predict the class using the MAPClassifier
#         predicted_class = map_classifier.predict(features)
#
#         # Compare the predicted class with the actual class
#         if predicted_class == actual_class:
#             correct_predictions += 1
#
#     # Calculate accuracy
#     acc = correct_predictions / total_predictions
#     return acc
#
#
# def multi_normal_pdf(x, mean, cov):
#     """
#     Calculate multi variable normal desnity function for a given x, mean and covarince matrix.
#
#     Input:
#     - x: A value we want to compute the distribution for.
#     - mean: The mean vector of the distribution.
#     - cov:  The covariance matrix of the distribution.
#
#     Returns the normal distribution pdf according to the given mean and var for the given x.
#     """
#     pdf = None
#     k = len(mean)  # Dimensionality of the data
#     x = np.array(x)
#     mean = np.array(mean)
#     cov = np.array(cov)
#
#     # Compute the normalization term
#     norm_term = 1 / (np.sqrt((2 * np.pi) ** k * np.linalg.det(cov)))
#
#     # Compute the exponent term
#     x_minus_mean = x - mean
#     inv_cov = np.linalg.inv(cov)
#     exponent_term = np.exp(-0.5 * np.dot(np.dot(x_minus_mean.T, inv_cov), x_minus_mean))
#
#     # Combine the terms to get the PDF
#     pdf = norm_term * exponent_term
#     return pdf
#
#
# class MultiNormalClassDistribution():
#
#     def __init__(self, dataset, class_value):
#         """
#         A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditional multi normal distribution.
#         The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
#
#         Input
#         - dataset: The dataset as a numpy array
#         - class_value : The class to calculate the parameters for.
#         """
#
#         # Extract the data for the specified class
#         self.data = dataset
#         class_data = dataset[dataset[:, -1] == class_value][:, :-1]
#         self.class_data = class_data
#
#         # Calculate the mean vector for the given class
#         self.mean = np.mean(class_data, axis=0)
#
#         # Calculate the covariance matrix for the given class
#         self.cov = np.cov(class_data.T)
#
#     def get_prior(self):
#         """
#         Returns the prior porbability of the class according to the dataset distribution.
#         """
#         prior = None
#
#         # Number of samples for the specific class
#         class_samples = self.class_data.shape[0]
#         total_samples = self.data.shape[0]
#
#         # Calculate the prior probability
#         prior = class_samples / total_samples
#
#         return prior
#
#     def get_instance_likelihood(self, x):
#         """
#         Returns the likelihood of the instance under the class according to the dataset distribution.
#         """
#         likelihood = None
#         likelihood = multi_normal_pdf(x, self.mean, self.cov)
#         return likelihood
#
#     def get_instance_posterior(self, x):
#         """
#         Returns the posterior porbability of the instance under the class according to the dataset distribution.
#         * Ignoring p(x)
#         """
#         posterior = None
#
#         prior = self.get_prior()
#         likelihood = self.get_instance_likelihood(x)
#
#         # The posterior is prior*likelihood
#         posterior = likelihood * prior
#
#         return posterior
#
#
# class MaxPrior():
#     def __init__(self, ccd0, ccd1):
#         """
#         A Maximum prior classifier.
#         This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
#         by the class that outputs the highest prior probability for the given instance.
#
#         Input
#             - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
#             - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
#         """
#         self.ccd0 = ccd0
#         self.ccd1 = ccd1
#
#     def predict(self, x):
#         """
#         Predicts the instance class using the 2 distribution objects given in the object constructor.
#
#         Input
#             - An instance to predict.
#         Output
#             - 0 if the posterior probability of class 0 is higher and 1 otherwise.
#         """
#         pred = None
#
#         prior0 = self.ccd0.get_prior()
#         prior1 = self.ccd1.get_prior()
#
#         # Compare the priors probabilities and predict the class
#         if prior0 > prior1:
#             pred = 0
#         else:
#             pred = 1
#
#         return pred
#
#
# class MaxLikelihood():
#     def __init__(self, ccd0, ccd1):
#         """
#         A Maximum Likelihood classifier.
#         This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
#         by the class that outputs the highest likelihood probability for the given instance.
#
#         Input
#             - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
#             - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
#         """
#         self.ccd0 = ccd0
#         self.ccd1 = ccd1
#
#     def predict(self, x):
#         """
#         Predicts the instance class using the 2 distribution objects given in the object constructor.
#
#         Input
#             - An instance to predict.
#         Output
#             - 0 if the posterior probability of class 0 is higher and 1 otherwise.
#         """
#         pred = None
#
#         likelihood0 = self.ccd0.get_instance_likelihood(x)
#         likelihood1 = self.ccd1.get_instance_likelihood(x)
#
#         # Compare the likelihoods probabilities and predict the class
#         if likelihood0 > likelihood1:
#             pred = 0
#         else:
#             pred = 1
#
#         return pred
#
#
# EPSILLON = 1e-6  # if a certain value only occurs in the test set, the probability for that value will be EPSILLON.
#
#
# class DiscreteNBClassDistribution():
#     def __init__(self, dataset, class_value):
#         """
#         A class which computes and encapsulate the relevant probabilites for a discrete naive bayes
#         distribution for a specific class. The probabilites are computed with laplace smoothing.
#
#         Input
#         - dataset: The dataset as a numpy array.
#         - class_value: Compute the relevant parameters only for instances from the given class.
#         """
#         # Extract the data for the specific class
#         self.data = dataset
#         class_data = dataset[dataset[:, -1] == class_value][:, :-1]
#         self.class_data = class_data
#         self.class_value = class_value
#         # Get the number of features
#         self.num_features = self.class_data.shape[1]
#         # Filed of a helper function
#         self.cond_probs = self.compute_conditional_probabilities()
#
#     def get_prior(self):
#         """
#         Returns the prior porbability of the class
#         according to the dataset distribution.
#         """
#         prior = None
#
#         # Number of samples for the specific class
#         class_samples_count = len(self.class_data)
#         # Total number of samples in the dataset
#         total_samples_count = len(self.data)
#         # Calculate the prior probability
#         prior = class_samples_count / total_samples_count
#
#         return prior
#
#     def get_instance_likelihood(self, x):
#         """
#         Returns the likelihood of the instance under
#         the class according to the dataset distribution.
#         """
#         likelihood = None
#         # Initialize likelihood to be 1 for the multiplication
#         likelihood = 1.0
#         # Iterate over each feature
#         for i in range(self.num_features):
#             feature_val = x[i]
#             feature_probs = self.cond_probs[i]
#             likelihood *= feature_probs.get(feature_val, 1 / (len(self.class_data) + len(feature_probs)))
#         return likelihood
#
#     def get_instance_posterior(self, x):
#         """
#         Returns the posterior porbability of the instance
#         under the class according to the dataset distribution.
#         * Ignoring p(x)
#         """
#         posterior = None
#
#         prior = self.get_prior()
#         likelihood = self.get_instance_likelihood(x)
#         # The posterior is prior*likelihood
#         posterior = likelihood * prior
#
#         return posterior
#
#     def compute_conditional_probabilities(self):
#         """
#         Helper function.
#         Computes conditional probabilities.
#         """
#         cond_probs = []  # List to store the conditional probabilities for each feature
#         # Iterate over each feature in the data
#         for i in range(self.num_features):
#             feature_vals = self.class_data[:, i]
#             unique_vals, counts = np.unique(feature_vals, return_counts=True)
#             feature_prob = {}
#             total_counts = len(feature_vals)
#
#             # Iterate over each unique value and its count
#             for val, count in zip(unique_vals, counts):
#                 # Compute the conditional probability
#                 feature_prob[val] = (count + 1) / (total_counts + len(unique_vals))
#             cond_probs.append(feature_prob)
#         return cond_probs
#
#
# class MAPClassifier_DNB():
#     def __init__(self, ccd0, ccd1):
#         """
#         A Maximum a posteriori classifier.
#         This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
#         by the class that outputs the highest posterior probability for the given instance.
#
#         Input
#             - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
#             - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
#         """
#         self.ccd0 = ccd0
#         self.ccd1 = ccd1
#
#     def predict(self, x):
#         """
#         Predicts the instance class using the 2 distribution objects given in the object constructor.
#
#         Input
#             - An instance to predict.
#         Output
#             - 0 if the posterior probability of class 0 is higher and 1 otherwise.
#         """
#         pred = None
#
#         posterior0 = self.ccd0.get_instance_posterior(x)
#         posterior1 = self.ccd1.get_instance_posterior(x)
#
#         # Compare the posterior probabilities and predict the class
#         if posterior0 > posterior1:
#             pred = 0
#         else:
#             pred = 1
#         return pred
#
#     def compute_accuracy(self, test_set):
#         """
#         Compute the accuracy of a given a testset using a MAP classifier object.
#
#         Input
#             - test_set: The test_set for which to compute the accuracy (Numpy array).
#         Ouput
#             - Accuracy = #Correctly Classified / #test_set size
#         """
#         acc = None
#         correct_predictions = 0
#
#         for instance in test_set:
#             # The feature vector is all columns except the last one
#             features = instance[:-1]
#             # The actual class label is the last column
#             actual_class = instance[-1]
#
#             # Predict the class using the MAPClassifier
#             predicted_class = self.predict(features)
#
#             # Compare the predicted class with the actual class
#             if predicted_class == actual_class:
#                 correct_predictions += 1
#
#         # Calculate accuracy
#         acc = correct_predictions / test_set.shape[0]
#         return acc
#
#
#
#
#
#
#
#
#
# #
# # #
# # # def multi_normal_pdf(x, mean, cov):
# # #     """
# # #     Calculate multi variable normal desnity function for a given x, mean and covarince matrix.
# # #
# # #     Input:
# # #     - x: A value we want to compute the distribution for.
# # #     - mean: The mean vector of the distribution.
# # #     - cov:  The covariance matrix of the distribution.
# # #
# # #     Returns the normal distribution pdf according to the given mean and var for the given x.
# # #     """
# # #     pdf = None
# # #     dim = len(mean)
# # #     cov_inv = np.linalg.pinv(cov)
# # #     cov_det = np.linalg.det(cov_inv)
# # #     norm_vector = 1 / ((2 * np.pi)**(dim / 2) * (cov_det)**(0.5))
# # #     exp = -0/5 * (x - mean)**2 / ()
# # #     pdf = (1 / (2 * np.pi)**(0.5) * np.linalg.det(cov)**(1/2)) * np.exp((-0.5) * (x - mean) * cov**-1 * (x - mean))
# # #
# # #     return pdf
# #
# # class MultiNormalClassDistribution():
# #
# #     def __init__(self, dataset, class_value):
# #         """
# #         A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
# #         The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
# #
# #         Input
# #         - dataset: The dataset as a numpy array
# #         - class_value : The class to calculate the parameters for.
# #         """
# #         ###########################################################################
# #         # TODO: Implement the function.                                           #
# #         ###########################################################################
# #         pass
# #         ###########################################################################
# #         #                             END OF YOUR CODE                            #
# #         ###########################################################################
# #
# #     def get_prior(self):
# #         """
# #         Returns the prior porbability of the class according to the dataset distribution.
# #         """
# #         prior = None
# #         ###########################################################################
# #         # TODO: Implement the function.                                           #
# #         ###########################################################################
# #         pass
# #         ###########################################################################
# #         #                             END OF YOUR CODE                            #
# #         ###########################################################################
# #         return prior
# #
# #     def get_instance_likelihood(self, x):
# #         """
# #         Returns the likelihood of the instance under the class according to the dataset distribution.
# #         """
# #         likelihood = None
# #         ###########################################################################
# #         # TODO: Implement the function.                                           #
# #         ###########################################################################
# #         pass
# #         ###########################################################################
# #         #                             END OF YOUR CODE                            #
# #         ###########################################################################
# #         return likelihood
# #
# #     def get_instance_posterior(self, x):
# #         """
# #         Returns the posterior porbability of the instance under the class according to the dataset distribution.
# #         * Ignoring p(x)
# #         """
# #         posterior = None
# #         ###########################################################################
# #         # TODO: Implement the function.                                           #
# #         ###########################################################################
# #         pass
# #         ###########################################################################
# #         #                             END OF YOUR CODE                            #
# #         ###########################################################################
# #         return posterior
# #
# # class MaxPrior():
# #     def __init__(self, ccd0 , ccd1):
# #         """
# #         A Maximum prior classifier.
# #         This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
# #         by the class that outputs the highest prior probability for the given instance.
# #
# #         Input
# #             - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
# #             - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
# #         """
# #         ###########################################################################
# #         # TODO: Implement the function.                                           #
# #         ###########################################################################
# #         pass
# #         ###########################################################################
# #         #                             END OF YOUR CODE                            #
# #         ###########################################################################
# #
# #     def predict(self, x):
# #         """
# #         Predicts the instance class using the 2 distribution objects given in the object constructor.
# #
# #         Input
# #             - An instance to predict.
# #         Output
# #             - 0 if the posterior probability of class 0 is higher and 1 otherwise.
# #         """
# #         pred = None
# #         ###########################################################################
# #         # TODO: Implement the function.                                           #
# #         ###########################################################################
# #         pass
# #         ###########################################################################
# #         #                             END OF YOUR CODE                            #
# #         ###########################################################################
# #         return pred
# #
# # class MaxLikelihood():
# #     def __init__(self, ccd0 , ccd1):
# #         """
# #         A Maximum Likelihood classifier.
# #         This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
# #         by the class that outputs the highest likelihood probability for the given instance.
# #
# #         Input
# #             - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
# #             - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
# #         """
# #         ###########################################################################
# #         # TODO: Implement the function.                                           #
# #         ###########################################################################
# #         pass
# #         ###########################################################################
# #         #                             END OF YOUR CODE                            #
# #         ###########################################################################
# #
# #     def predict(self, x):
# #         """
# #         Predicts the instance class using the 2 distribution objects given in the object constructor.
# #
# #         Input
# #             - An instance to predict.
# #         Output
# #             - 0 if the posterior probability of class 0 is higher and 1 otherwise.
# #         """
# #         pred = None
# #         ###########################################################################
# #         # TODO: Implement the function.                                           #
# #         ###########################################################################
# #         pass
# #         ###########################################################################
# #         #                             END OF YOUR CODE                            #
# #         ###########################################################################
# #         return pred
# #
# # EPSILLON = 1e-6 # if a certain value only occurs in the test set, the probability for that value will be EPSILLON.
# #
# # class DiscreteNBClassDistribution():
# #     def __init__(self, dataset, class_value):
# #         """
# #         A class which computes and encapsulate the relevant probabilites for a discrete naive bayes
# #         distribution for a specific class. The probabilites are computed with laplace smoothing.
# #
# #         Input
# #         - dataset: The dataset as a numpy array.
# #         - class_value: Compute the relevant parameters only for instances from the given class.
# #         """
# #         ###########################################################################
# #         # TODO: Implement the function.                                           #
# #         ###########################################################################
# #         pass
# #         ###########################################################################
# #         #                             END OF YOUR CODE                            #
# #         ###########################################################################
# #
# #     def get_prior(self):
# #         """
# #         Returns the prior porbability of the class
# #         according to the dataset distribution.
# #         """
# #         prior = None
# #         ###########################################################################
# #         # TODO: Implement the function.                                           #
# #         ###########################################################################
# #         pass
# #         ###########################################################################
# #         #                             END OF YOUR CODE                            #
# #         ###########################################################################
# #         return prior
# #
# #     def get_instance_likelihood(self, x):
# #         """
# #         Returns the likelihood of the instance under
# #         the class according to the dataset distribution.
# #         """
# #         likelihood = None
# #         ###########################################################################
# #         # TODO: Implement the function.                                           #
# #         ###########################################################################
# #         pass
# #         ###########################################################################
# #         #                             END OF YOUR CODE                            #
# #         ###########################################################################
# #         return likelihood
# #
# #     def get_instance_posterior(self, x):
# #         """
# #         Returns the posterior porbability of the instance
# #         under the class according to the dataset distribution.
# #         * Ignoring p(x)
# #         """
# #         posterior = None
# #         ###########################################################################
# #         # TODO: Implement the function.                                           #
# #         ###########################################################################
# #         pass
# #         ###########################################################################
# #         #                             END OF YOUR CODE                            #
# #         ###########################################################################
# #         return posterior
# #
# #
# # class MAPClassifier_DNB():
# #     def __init__(self, ccd0 , ccd1):
# #         """
# #         A Maximum a posteriori classifier.
# #         This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
# #         by the class that outputs the highest posterior probability for the given instance.
# #
# #         Input
# #             - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
# #             - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
# #         """
# #         ###########################################################################
# #         # TODO: Implement the function.                                           #
# #         ###########################################################################
# #         pass
# #         ###########################################################################
# #         #                             END OF YOUR CODE                            #
# #         ###########################################################################
# #
# #     def predict(self, x):
# #         """
# #         Predicts the instance class using the 2 distribution objects given in the object constructor.
# #
# #         Input
# #             - An instance to predict.
# #         Output
# #             - 0 if the posterior probability of class 0 is higher and 1 otherwise.
# #         """
# #         pred = None
# #         ###########################################################################
# #         # TODO: Implement the function.                                           #
# #         ###########################################################################
# #         pass
# #         ###########################################################################
# #         #                             END OF YOUR CODE                            #
# #         ###########################################################################
# #         return pred
# #
# #     def compute_accuracy(self, test_set):
# #         """
# #         Compute the accuracy of a given a testset using a MAP classifier object.
# #
# #         Input
# #             - test_set: The test_set for which to compute the accuracy (Numpy array).
# #         Ouput
# #             - Accuracy = #Correctly Classified / #test_set size
# #         """
# #         acc = None
# #         ###########################################################################
# #         # TODO: Implement the function.                                           #
# #         ###########################################################################
# #         pass
# #         ###########################################################################
# #         #                             END OF YOUR CODE                            #
# #         ###########################################################################
# #         return acc
# #
# #
