import numpy as np
import matplotlib.pyplot as plt
import queue 

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0

    # Get the labels
    labels = data[:, -1]

    # Compute gini
    sigma = 0.0
    classes = np.unique(labels)
    for cls in classes:
        p_cls = np.sum(labels == cls) / len(labels)
        sigma += p_cls ** 2

    gini = 1 - sigma
    return gini


def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0

    # Get the lables
    labels = data[:, -1]

    # Compute Entropy
    classes, counts = np.unique(labels, return_counts=True)
    for count in counts:
        propobility = count / len(labels)
        entropy -= propobility * np.log2(propobility)

    return entropy


class DecisionNode:

    def __init__(self, data, impurity_func, feature=-1, depth=0, chi=1, max_depth=1000, gain_ratio=False):

        self.data = data  # the relevant data for the node
        self.feature = feature  # column index of criteria being tested
        self.pred = self.calc_node_pred()  # the prediction of the node
        self.depth = depth  # the current depth of the node
        self.children = []  # array that holds this nodes children
        self.children_values = []  # holds the value of the feature associated with the children (list)
        self.terminal = False  # determines if the node is a leaf
        self.chi = chi
        self.max_depth = max_depth  # the maximum allowed depth of the tree
        self.impurity_func = impurity_func
        self.gain_ratio = gain_ratio
        self.feature_importance = 0

    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        # Get the unique labels and their counts
        unique_labels, count = np.unique(self.data.T[-1], return_counts=True)
        dict_counter = dict(zip(unique_labels, count))
        # Check if the dictionary is empty
        if not dict_counter:
            return pred  # Return None if the dictionary is empty
        # Choose the label with the highets count as the prediction
        pred = max(dict_counter, key=dict_counter.get)
        return pred

    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)

    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.

        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in
        self.feature_importance
        """
        # Check if there is no feature importance
        if self.feature == -1:
            self.feature_importance = 0.0

        # Calculate the wighted impurity decrease
        imp_before_split = self.impurity_func(self.data)
        imp_after_split = 0.0

        # Calculate impurity after splitting
        for child in self.children:
            imp_after_split += (len(child.data) / n_total_sample) * self.impurity_func(child.data)

        impurity_decrease = imp_before_split - imp_after_split
        self.feature_importance = impurity_decrease

    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting
                  according to the feature values.
        """
        goodness = 0
        groups = {}  # groups[feature_value] = data_subset

        # If gain_ratio is True, use entropy as impurity function.
        if self.gain_ratio:
            impurity_func = calc_entropy

        # Initialize variables for calculation.
        split_in_information = 0
        impurity_data = self.impurity_func(self.data)
        impurity_attribute = 0
        unique_values = np.unique(self.data[:, feature])

        # Calculate split information and impurity for each feature value.
        for val in unique_values:
            subgroup = self.data[self.data[:, feature] == val]
            groups[val] = subgroup
            size_attribute = len(groups[val]) / len(self.data)
            split_in_information += size_attribute * np.log2(size_attribute)
            impurity_attribute += size_attribute * self.impurity_func(groups[val])

        # Calculate goodness of split based on gain_ratio flag.
        if not self.gain_ratio:
            goodness = impurity_data - impurity_attribute
        else:
            information_gain = impurity_data - impurity_attribute

            # Avoid division by zero.
            if split_in_information == 0:
                return 0, groups

            # Compute goodness of split using gain ratio formula.
            split_in_information *= -1
            goodness = information_gain / split_in_information

        return goodness, groups

    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """
        # Check if the tree depth is less than the maximal depth
        if self.depth == self.max_depth:
            self.terminal = True
            return

        # Finding the best feature to split by and take its data
        self.feature = index_best_feature(self, self.data, self.impurity_func, self.gain_ratio)
        _, subdata = self.goodness_of_split(self.feature)

        # Checking if the chi pruning condition is met and if we will have more than 1 child.
        if len(subdata) > 1 and check_chi(self, subdata, self.chi):
            for key, group in subdata.items():
                child = DecisionNode(group, self.impurity_func, feature=self.feature, depth=self.depth + 1,
                                     chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio)
                self.add_child(child, key)
        else:
            self.terminal = True
            

def index_best_feature(node, data, impurity_func, gain_ratio):
    """
    Helper function.
    Finds the index of the best feature to split by.

    Args:
        - node: An instance of the DecisionNode class.
        - data: The dataset where the last column holds the labels.
        - impurity_func: The impurity function to be used as the splitting criterion.
        - gain_ratio: Flag indicating whether to use goodness of split or gain ratio.

    Returns:
        The index of the feature that provides the best split.
    """
    # Check if the data array is empty
    if len(data) == 0:
        return None
        
    # Initialize variables to find the index of the best feature to split by.
    best_feature = None
    best_feature_goodness = float('-inf')

    # Iterate over all features and calculate the goodness of split for each.
    for index in range(len(data[0]) - 1):
        current_feature_goodness, _ = node.goodness_of_split(index)

        # Update the index of the feature that gives the highest goodness of split.
        if current_feature_goodness > best_feature_goodness:
            best_feature_goodness = current_feature_goodness
            best_feature = index
    return best_feature
    
def check_chi(node, subdata, chi):
    """
    Helper function.
    Checks if the conditions for performing chi pruning are met.

    Args:
        - node: The decision tree node.
        - subdata: The data obtained after calculating the goodness of split using the best feature.
        - chi: The chi value obtained during tree construction.

    Returns:
        True if the calculated chi value is greater than or equal to the chi value from the chi table, False otherwise.
    """
    # If the chi value is 1, no need to perform chi pruning.
    if chi == 1:
        return True

    # Calculate the chi value using the formula and compare it with the value from the chi table.
    chi_val = chi_square_compute(node.data, subdata)
    deg_of_freedom = len(subdata) - 1
    chi_val_from_table = chi_table[deg_of_freedom][chi]
    return chi_val >= chi_val_from_table

def chi_square_compute(data, subdata):
    """
    Helper function.
    Calculates the chi square statistic according to the formula.

    Args:
    - data: The entire dataset where the last column holds the labels.
    - subdata: A dictionary containing subsets of data after splitting.

    Returns:
    - chi_square: The chi square statistic of the dataset.
    """
    # Initialize the chi square statistic.
    chi_square = 0.0

    # Get the size of the entire dataset.
    total_size = len(data[:, -1])

    # Get the count of each label in the entire dataset.
    total_label_count = dict_label_number(data[:, -1])

    # Iterate over each feature value and its corresponding subset.
    for feature_val, sub in subdata.items():
        # Get the size of the subset.
        sub_size = len(sub)
        
        # Get the count of each label in the subset.
        sub_label_count = dict_label_number(sub[:, -1])
        
        # Iterate over each label and its count in the entire dataset.
        for label, count in total_label_count.items():
            # Calculate the expected count based on the subset size and label proportion in the entire dataset.
            expected = sub_size * (count / total_size)
            
            # Get the observed count of the label in the subset.
            observed = sub_label_count.get(label, 0)
            
            # Calculate the contribution to the chi square statistic.
            chi_square += ((observed - expected) ** 2) / expected
    
    return chi_square
    
def dict_label_number(labels):
    """
    Helper function.
    Creates a dictionary of label counts.

    Args:
    - labels: Array of labels.

    Returns:
    - label_counts: Dictionary where each key is a unique label and the value is the count of occurrences.
    """
    # Count the occurrences of each unique label in the array of labels.
    label_counts = {label: np.sum(labels == label) for label in np.unique(labels)}
    
    return label_counts


class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data  # the relevant data for the tree
        self.impurity_func = impurity_func  # the impurity function to be used in the tree
        self.chi = chi
        self.max_depth = max_depth  # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio  #
        self.root = None  # the root node of the tree

    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset.
        You are required to fully grow the tree until all leaves are pure
        or the goodness of split is 0.

        This function has no return value
        """
        # Start the tree from the root
        self.root = DecisionNode(data=self.data, impurity_func=self.impurity_func, chi=self.chi,
                                    max_depth=self.max_depth, gain_ratio=self.gain_ratio)
        # Create a queue to insert the nodes into the tree by order
        q = queue.Queue()
        q.put(self.root)

        # Run as long as there is a node that can be inserted to the tree
        while not q.empty():
            node = q.get()
            # Check if the node is terminal, then no further splitting is needed
            if node.terminal:
                continue

            # Check if the data is arranged
            if len(np.unique(node.data)) == 1:
                node.terminal = True
                continue
            
            # Split the node
            node.split()
            # Check if we didn't find a feature to split
            if node.feature is None:
                node.terminal = True
                continue

            # Fill the que with the node's children
            for child in node.children:
                q.put(child)

    def predict(self, instance):
        """
        Predict a given instance

        Input:
        - instance: an row vector from the dataset. Note that the last element
                    of this vector is the label of the instance.

        Output: the prediction of the instance.
        """
        pred = None
        # Start traversal from the root node
        node = self.root
        # Flag to check if a matching child node is found
        found_child = True

        # Traverse the tree until a terminal node is reached or a matching child node is found
        while not node.terminal and found_child:
            found_child = False
            # Iterate over each child of the current node
            for child in node.children:
                # Check if the feature value of the child node matches the instance
                if (child.data[:, node.feature] == instance[node.feature]).all():
                    # Move to the matching child node
                    node = child
                    found_child = True
                    break  # Exit the for loop after moving to the child node

        # Return the predicted class label of the instance
        pred = node.pred
        return pred

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset

        Input:
        - dataset: the dataset on which the accuracy is evaluated

        Output: the accuracy of the decision tree on the given dataset (%).
        """
        accuracy = 0
        correct_pred = 0

        # Iterate over each instance in the dataset and check if the prediction is good
        for instance in dataset:
            pred_label = self.predict(instance)
            actual_label = instance[-1]

            if pred_label == actual_label:
                correct_pred += 1

        # Calculate the accuracy
        accuracy = (correct_pred / (len(dataset))) * 100
        return accuracy

    def depth(self):
        return self.root.depth()


def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. On a single plot, draw the training and testing accuracy
    as a function of the max_depth.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels

    Output: the training and validation accuracies per max depth
    """
    training = []
    validation = []
    root = None
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        tree = DecisionTree(data=X_train, impurity_func=calc_entropy, max_depth=max_depth, gain_ratio=True)
        tree.build_tree() 
        train = tree.calc_accuracy(X_train)
        valid = tree.calc_accuracy(X_validation)
        training.append(train)
        validation.append(valid)

    return training, validation


def chi_pruning(X_train, X_test):
    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels

    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_validation_acc = []
    depth = []
    for chi_value in [0.5, 0.25, 0.1, 0.05, 0.0001]:
        tree = DecisionTree(data=X_train, impurity_func=calc_entropy, chi=chi_value)
        tree.build_tree()
        training_accuracy = tree.calc_accuracy(X_train)
        validation_accuracy = tree.calc_accuracy(X_test)
        chi_training_acc.append(training_accuracy)
        chi_validation_acc.append(validation_accuracy)
        depth.append(tree.root.depth) 
    return chi_training_acc, chi_validation_acc, depth


def count_nodes(node):
    """
    Count the number of node in a given tree

    Input:
    - node: a node in the decision tree.

    Output: the number of node in the tree.
    """
    # Base case: if the node is None, return 0
    if node is None:
        return 0

    # Initialize the node count to 1 (for the current node)
    n_nodes = 1

    # Recursively count nodes in the children of the current node
    if not node.terminal:
        for child in node.children:
            n_nodes += count_nodes(child)

    return n_nodes
