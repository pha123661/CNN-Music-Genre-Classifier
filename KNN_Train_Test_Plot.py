import matplotlib.pyplot as plt
from Hyper_parameters import HyperParams
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from myDataLoader import get_ndarrays
from sklearn.model_selection import GridSearchCV
import numpy as np
from CNN_Train_Test_Plot import *

def main():
    x_train, y_train, x_test, y_test, x_valid, y_valid = get_ndarrays()
    # shape = (samples, 512*2*2)
    for k in [5, 7, 10, 20, 40, 100]:
        KNNclf = KNeighborsClassifier(n_jobs=-1, n_neighbors=k) # get by grid search
        KNNclf.fit(x_train, y_train)
        y_pred = KNNclf.predict(x_valid)
        # print(confusion_matrix(y_test, y_pred))
        print(f"k = {k}")
        print(f"{accuracy_score(y_valid, y_pred):.6f} {recall_score(y_valid, y_pred, average='macro'):.6f} {precision_score(y_valid, y_pred, average='macro'):.6f}")
        
    KNNclf = KNeighborsClassifier(n_jobs=-1, n_neighbors=5) # get by grid search
    KNNclf.fit(x_train, y_train)
    y_pred = KNNclf.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    print(f"testing")
    print(f"{accuracy_score(y_test, y_pred):.6f} {recall_score(y_test, y_pred, average='macro'):.6f} {precision_score(y_test, y_pred, average='macro'):.6f}")
        
if __name__ == '__main__':
    main()