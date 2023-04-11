class Node:
    def __init__(self,feature=None,thresh=None,left=None,right=None,*,value=None):
        self.feature
        self.thresh
        self.left
        self.right
        self.value=None
    def is_leaf(self):
        return self.value is not None
        
        



class Tree:
    def __init__(self,min_smaples_split=2,max_depth=100,n_features=None):
        self.min_smaples_split =min_smaples_split
        self.max_depth= max_depth
        self.n_features=n_features
        pass

    def fit(self,x,y):
        self.n_features = x.shape[1] if not self.n_features else min(x.shape[1], self.n_features)
        self.root = self._growtree(x,y)

        pass
    def _growtree(self):
        pass

    def predict(self):
        pass