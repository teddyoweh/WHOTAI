
import numpy as np
import re
import pandas as pd
from collections import Counter

class TokenizerGen:
    def __init__(self):
        self.word_count = Counter()
    
    def fit(self, df):
        texts = df.values.flatten()
        for text in texts:
            words = self.tokenize(text)
            self.word_count.update(words)
    
    def tokenize(self, text):
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return words
    
    def transform(self, df):
        X_transformed = []
        for col in df.columns:
            texts = df[col].values
            col_transformed = []
            for text in texts:
                words = self.tokenize(text)
                x = [self.word_count[word] for word in words]
                col_transformed.append(x)
            X_transformed.append(col_transformed)
        X_transformed = pd.DataFrame(X_transformed).T
        X_transformed.columns = df.columns
        return X_transformed
    
class Logistic:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.m, self.n = X.shape
        self.k = len(np.unique(y))
        self.W = np.zeros((self.n, self.k))
        self.b = np.zeros((1, self.k))
        
        for i in range(self.num_iterations):
            z = np.dot(X, self.W) + self.b
            y_pred = self.softmax(z)
            dW = (1 / self.m) * np.dot(X.T, (y_pred - self.one_hot(y)))
            db = (1 / self.m) * np.sum(y_pred - self.one_hot(y), axis=0, keepdims=True)
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db
    
    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
    
    def predict(self, X):
        z = np.dot(X, self.W) + self.b
        y_pred = self.softmax(z)
        return np.argmax(y_pred, axis=1)
    
    def one_hot(self, y):
        return np.eye(self.k)[y]


class Pipeline:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
    
    def preprocess(self, texts):
        tokenized = self.tokenizer(texts, padding=True, truncation=True, return_tensors='tf')
        return tokenized
    
    def predict(self, texts):
        preprocessed = self.preprocess(texts)
        predictions = self.model.predict(preprocessed)
        return np.argmax(predictions, axis=1)


class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2, min_impurity_decrease=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.tree = {}

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        return [self._traverse_tree(x, self.tree) for x in X]

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_samples < self.min_samples_split or n_labels == 1):
            leaf_value = self._leaf_value(y)
            return {'type': 'leaf', 'value': leaf_value}

        best_feature, best_threshold = self._best_criteria(X, y)

        if best_feature is None:
            leaf_value = self._leaf_value(y)
            return {'type': 'leaf', 'value': leaf_value}

        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold

        left_tree = self._grow_tree(X[left_indices], y[left_indices], depth+1)
        right_tree = self._grow_tree(X[right_indices], y[right_indices], depth+1)

        return {'type': 'split',
                'feature_index': best_feature,
                'threshold': best_threshold,
                'left': left_tree,
                'right': right_tree}

    def _best_criteria(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None

        impurity_parent = self._impurity(y)

        for feature_idx in range(X.shape[1]):
            feature_values = X[:, feature_idx]

            thresholds = np.unique(feature_values)

            for threshold in thresholds:
                left_indices = feature_values < threshold
                right_indices = feature_values >= threshold

                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue

                impurity_left = self._impurity(y[left_indices])
                impurity_right = self._impurity(y[right_indices])

                gain = impurity_parent - (len(y[left_indices]) / len(y) * impurity_left +
                                          len(y[right_indices]) / len(y) * impurity_right)

                if gain > best_gain and gain > self.min_impurity_decrease:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _impurity(self, y):
        _, counts = np.unique(y, return_counts=True)
        impurity = 1 - np.sum(np.square(counts / len(y)))
        return impurity

    def _leaf_value(self, y):
        _, counts = np.unique(y, return_counts=True)
        return np.argmax(counts)

    def _traverse_tree(self, x, tree):
        if tree['type'] == 'leaf':
            return tree['value']

        feature_value = x[tree['feature_index']]

        if feature_value < tree['threshold']:
            return self._traverse_tree(x, tree['left'])
        else:
            return self._traverse_tree(x, tree['right'])
