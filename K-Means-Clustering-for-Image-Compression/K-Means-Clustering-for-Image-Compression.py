import numpy as np

def get_random_centroids(X, k):
    '''
    Each centroid is a point in RGB space (color) in the image. 
    This function should uniformly pick `k` centroids from the dataset.
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids. 
    Notice we are flattening the image to a two dimentional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array. 
    '''

    # choose k random rows from the data set
    random_centroids = np.random.choice(X.shape[0], k, replace=False)
    centroids = X[random_centroids, :]

    return np.asarray(centroids).astype(np.float32)


def lp_distance(X, centroids, p=2):
    '''
    Inputs: 
    A single image of shape (num_pixels, 3)
    The centroids (k, 3)
    The distance parameter p

    output: numpy array of shape `(k, num_pixels)` thats holds the distances of 
    all points in RGB space from all centroids
    '''
    # initialize empty array
    k = len(centroids)
    distances = np.zeros((k, X.shape[0]))

    for i in range(k):
        # calculate the distance from each pixel to the i'th centroid
        distances[i] = np.sum(np.abs(X - centroids[i])**p, axis=1)**(1/p)

    return distances

def kmeans(X, k, p ,max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    centroids = get_random_centroids(X, k)
    cur_centroids = np.zeros_like(centroids)
    classes = np.zeros(X.shape[0])

    for iter in range(max_iter):
        # compute the distance
        distances = lp_distance(X, centroids, p)

        # checks the distance matrix and assings each pixel to the index of the closest centroid out of the k
        classes = np.argmin(distances, axis=0)

        # update centroids
        for i in range(k):
            if np.any(classes == i):
                centroids[i] = np.mean(X[classes == i], axis=0)

        # check if converged and there is nothing left to update
        if np.all(centroids == cur_centroids):
            break

        cur_centroids = centroids.copy()


    return centroids, classes

def kmeans_pp(X, k, p ,max_iter=100):
    """
    Your implenentation of the kmeans++ algorithm.
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """

    # choose random centroid uniformly
    num_pixels = X.shape[0]

    # choose uniformly random first centroid
    centroids = [X[np.random.choice(num_pixels)]]

    # choose the remaining k-1 centroids
    for i in range(1, k):
        # compute the distance from each point to the nearest chosen centroid
        distances = lp_distance(X, centroids, p)
        min_distance = np.min(distances, axis=0)

        # compute probability distribution for the next centroid
        squared_distance = min_distance**2
        probabilities = squared_distance / np.sum(squared_distance)

        # choose new centroid with the weighted distribution
        next_centroid_index = np.random.choice(num_pixels, p=probabilities)
        centroids.append(X[next_centroid_index])

    centroids = np.array(centroids)
    centroids, classes = kmeans(X, k, p, max_iter)
    return centroids, classes
