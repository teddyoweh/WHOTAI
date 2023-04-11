import numpy as np
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

import pickle

 

class Utils:
    def __init__(self) -> None:
        pass
    @staticmethod
    def save_object(obj, filename):
  
        with open(filename, 'wb') as file:
            pickle.dump(obj, file)
    @staticmethod
    def load_object(filename):
        with open(filename, 'rb') as file:
            obj = pickle.load(file)
        return obj
    @staticmethod
    def findkey(d, value):
      sorted_items = sorted(d.items(), key=lambda item: item[1])

      def trinary_search(left, right):
        if left <= right:
            one_third = left + (right - left) // 3
            two_third = right - (right - left) // 3
            one_third_value = sorted_items[one_third][1]
            two_third_value = sorted_items[two_third][1]

            if one_third_value == value:
                return sorted_items[one_third][0]
            elif two_third_value == value:
                return sorted_items[two_third][0]
            elif value < one_third_value:
                return trinary_search(left, one_third - 1)
            elif value > two_third_value:
                return trinary_search(two_third + 1, right)
            else:
                return trinary_search(one_third + 1, two_third - 1)

        return None

      return trinary_search(0, len(sorted_items) - 1)