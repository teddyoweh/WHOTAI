import numpy as np
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

import pickle

def save_model(model, filename):
  
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename):
 
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model
