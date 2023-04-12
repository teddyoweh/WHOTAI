
import numpy as np
from collections import Counter
import pandas as pd
import pickle
from multiprocessing import Pool ,cpu_count

from utils import Utils
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.n_features=n_features
        self.root=None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.root = self._grow_tree(X, y)
       

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

 
        if (depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # find the best split
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feature, best_thresh, left, right)


    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
        
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold


    def _information_gain(self, y, X_column, threshold):
 
        parent_entropy = self._entropy(y)

   
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
 
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

 
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p>0])


    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
        


class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_feature=None):
        self.n_trees = n_trees
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.n_features=n_feature
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth,
                            min_samples_split=self.min_samples_split,
                            n_features=self.n_features)
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions
    

class CleanTokenize:
    def __init__(self,data) -> None:
        self.data = data
        pass
    @property 
    def df(self)->pd.DataFrame:
        ins = self.data.drop(columns=['id'])
        arrs = [__ for _ in ins.values for __ in _]
        self.tokens  = {_:i for i,_ in enumerate(set(arrs)) }
        insdata = []
        for i in ins.values:
            insdata.append([self.tokens[_] for _ in i])
        return pd.DataFrame(data=insdata,columns=ins.columns)
    
    
class PreTrainedWhotAI(CleanTokenize):
    def __init__(self,data,model=RandomForest()):
        super().__init__(data)
        self.wdf = self.df.drop_duplicates()
        self.cards,self.ActionCard = self.wdf.drop(columns=['Action']),self.wdf['Action']
        self.model = model
        self.model.fit(self.cards.values,self.ActionCard.values)
    def save(self,modelname,tokenname):
        Utils.save_object(self.model,modelname)
        Utils.save_object(self.tokens,tokenname)

    def predict(self,cards,played):
        cards.sort()
        card1,card2,card3,card4 = cards

        return Utils.findkey(self.tokens,self.model.predict([[self.tokens[card1],self.tokens[card2],self.tokens[card3],self.tokens[card4],self.tokens[played]]])[0])

    
    
class PostTrainedWhotAI(object):
    def __init__(self, model,tokens) -> None:
        self.model = Utils.load_object(model)
        self.tokens = Utils.load_object(tokens)
    def predict(self,cards,played):
        cards.sort()
        card1,card2,card3,card4 = cards
        #print(self.model.predict([[20,19,34,49,20]]))
        #print(self.tokens[card1],self.tokens[card2],self.tokens[card3],self.tokens[card4],self.tokens[played])
        return self.model.predict([self.tokens[card1],self.tokens[card2],self.tokens[card3],self.tokens[card4],self.tokens[played]])
        #return Utils.findkey(self.tokens,[0])

def test_postmode():
    model = PostTrainedWhotAI('whotmodel','whottokens')
    print(model.predict(['circle 1','circle 2','circle 3','circle 4'],'circle 1'))
    
def test_premodel():
    whot = PreTrainedWhotAI(pd.read_csv('test.csv',nrows=1000))
    whot.save('whotmodel','whottokens')
    print('Finished Training')




def parrellize_model():
    def train_model(data):
        whot = PreTrainedWhotAI(data)
        whot.save('whotmodel', 'whottokens')
        print('Finished Training')

    if __name__ == '__main__':
        data = pd.read_csv('train.csv')
        
        num_processes = 4
        data_splits = np.array_split(data, num_processes)
        
    
        if len(data_splits) != num_processes:
            raise ValueError("Number of data splits does not match number of processes")
        
        with Pool(processes=num_processes) as pool:
            try:
                pool.map(train_model, data_splits)
            except IndexError:
        
                print([len(split) for split in data_splits])
                raise

    