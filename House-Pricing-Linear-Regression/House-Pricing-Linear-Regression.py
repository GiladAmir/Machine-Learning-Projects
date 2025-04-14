###### Your ID ######
# ID1: 208614883
# ID2: 313372997
#####################

# imports 
import numpy as np
import pandas as pd

def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    X = (X - X.min()) / (X.max() - X.min())
    y = (y - y.min()) / (y.max() - y.min())
    return X, y

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    bias_col = np.ones((X.shape[0], 1))
    X = np.concatenate([bias_col, X], axis = 1)
    return X

def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """
    
    J = 0  # We use J for the cost.
    m = len(y)  # Number of training examples
    predictions = X.dot(theta)
    errors = predictions - y
    squared_errors = errors**2
    J = (1 / (2 * m)) * np.sum(squared_errors)
    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """


    theta = theta.copy()  # optional: ensures theta outside the function will not change
    J_history = []  # List to save the cost value at every iteration
    clip_threshold = 1.0  # Hard-coded threshold for clipping gradients


    for i in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = (1 / len(y)) * X.T.dot(errors)

        if np.linalg.norm(gradient) > clip_threshold:
            gradient = gradient * (clip_threshold / np.linalg.norm(gradient))
        theta -= alpha * gradient

        cost = compute_cost(X, y, theta)
        J_history.append(cost)
        # Early exit if there is a NaN
        if np.isnan(cost):
            break

    return theta, J_history



def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """

    pinv_theta = []
    transposed = np.transpose(X)

    XtX = np.dot(transposed, X)
    if np.linalg.det(XtX) != 0:

        pinv_XtX = np.linalg.inv(XtX)
        pinv = np.dot(pinv_XtX, transposed)
        pinv_theta = np.dot(pinv, y)
        return pinv_theta
    else:
        return None



def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """

    theta = theta.copy()  # optional: ensures theta outside the function will not change
    J_history = []  # List to save the cost value at every iteration
    clip_threshold = 1.0

    for i in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = (1 / len(y)) * X.T.dot(errors)
        # Gradient clipping
        if np.linalg.norm(gradient) > clip_threshold:
            gradient = gradient * (clip_threshold / np.linalg.norm(gradient))
        theta -= alpha * gradient

        # Compute and record the cost
        cost = compute_cost(X, y, theta)
        J_history.append(cost)

        # Early exit if NaN detected
        if np.isnan(cost):
            break

        # Check if the improvement in loss is less than the threshold to stop early
        if i > 0 and abs(J_history[i - 1] - J_history[i]) < 1e-8:
            break

    return theta, J_history

def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {} # {alpha_value: validation_loss}
    init_theta = np.random.random(size = X_train.shape[1]) * 0.01 #initialize thetas vector
    for alpha in alphas:
        theta, J_history = efficient_gradient_descent(X_train, y_train, init_theta, alpha, iterations)
        alpha_dict[alpha] = compute_cost(X_val, y_val, theta)

    return alpha_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to
    select the most relevant features for a predictive model. The objective
    of this algorithm is to improve the model's performance by identifying
    and using only the most relevant features, potentially reducing overfitting,
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part.

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    num_features = 5

    while len(selected_features) < num_features:
        feature_scores = {}  # keep track of feature scores

        # Iterate over each feature that is not yet selected
        for i in range(X_train.shape[1]):
            if i not in selected_features:
                # Temporarily include this feature to the selected list
                current_features = selected_features + [i]

                # Extract the columns for the current feature set
                X_train_temp = X_train[:, current_features]
                X_val_temp = X_val[:, current_features]

                # Apply the bias trick
                X_train_temp = apply_bias_trick(X_train_temp)
                X_val_temp = apply_bias_trick(X_val_temp)

                # Initialize theta for this subset of features
                theta_rand = np.random.rand(X_train_temp.shape[1])
                theta = theta_rand.copy()

                # Use gradient descent to train the model on the training subset
                theta, J_history = efficient_gradient_descent(X_train_temp, y_train, theta, best_alpha, iterations)

                # Compute the cost on the validation set
                cost = compute_cost(X_val_temp, y_val, theta)

                # Store the cost of  this feature
                feature_scores[i] = cost

        # Find the feature that when added, results in the lowest validation cost
        best_feature = min(feature_scores, key=feature_scores.get)
        selected_features.append(best_feature)

    return selected_features





def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()

    # Add square of each feature
    for col in df.columns:
        df_poly[f'{col}^2'] = df[col] ** 2

    # Add interaction terms for each pair of different features
    for i in range(len(df.columns)):
        for j in range(i + 1, len(df.columns)):
            col1 = df.columns[i]
            col2 = df.columns[j]
            df_poly[f'{col1}*{col2}'] = df[col1] * df[col2]

    return df_poly