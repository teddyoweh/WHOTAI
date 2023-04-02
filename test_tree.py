import numpy as np

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        
    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))
        self.tree_ = self._grow_tree(X, y)
        
    def predict(self, X):
        return [self._predict(inputs) for inputs in X]
    
    def _predict(self, inputs):
        node = self.tree_
        while node.value is None:
            if inputs[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value
    
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        if (self.max_depth is not None and depth >= self.max_depth) or n_labels == 1:
            return Node(value=self._most_common_label(y))
        
        feature_indices = np.random.choice(n_features, n_features, replace=False)
        
        best_feature, best_threshold = self._best_criteria(X, y, feature_indices)
        
        left_indices = X[:, best_feature] <= best_threshold
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[~left_indices], y[~left_indices]
        
        left = self._grow_tree(X_left, y_left, depth + 1)
        right = self._grow_tree(X_right, y_right, depth + 1)
        
        return Node(best_feature, best_threshold, left, right)
    
    def _best_criteria(self, X, y, feature_indices):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feature_index in feature_indices:
            X_column = X[:, feature_index]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_index
                    split_thresh = threshold
        return split_idx, split_thresh
    
    def _information_gain(self, y, X_column, split_thresh):
        parent_entropy = self._entropy(y)
        left_indices = X_column <= split_thresh
        right_indices = X_column > split_thresh
        n = len(y)
        n_l, n_r = len(y[left_indices]), len(y[right_indices])
        e_l, e_r = self._entropy(y[left_indices]), self._entropy(y[right_indices])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        ig = parent_entropy - child_entropy
        return ig
    
    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = sum(probabilities * -np.log2(probabilities))
        return entropy
    
    def _most_common_label(self, y):
        _, counts = np.unique(y, return_counts=True)
        return np.argmax(counts)
