import matplotlib.pyplot as plt
from Hyper_parameters import HyperParams
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from myDataLoader import get_ndarrays
from sklearn.model_selection import GridSearchCV
import numpy as np
from CNN_Train_Test_Plot import *

def main():
    x_train, y_train, x_test, y_test, x_valid, y_valid = get_ndarrays()
    # shape = (samples, 512*2*2)
    for loss in ('hinge', 'squared_hinge'):
        SVM = LinearSVC(loss=loss, max_iter=100000) # get by grid search
        SVM.fit(x_train, y_train)
        y_pred = SVM.predict(x_valid)
        # print(confusion_matrix(y_test, y_pred))
        print("loss:",loss)
        print(f"{accuracy_score(y_valid, y_pred):.6f} {recall_score(y_valid, y_pred, average='macro'):.6f} {precision_score(y_valid, y_pred, average='macro'):.6f}")
        
    SVM = LinearSVC(loss='hinge', max_iter=100000) # get by grid search
    SVM.fit(x_train, y_train)
    y_pred = SVM.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    print(f"testing")
    print(f"{accuracy_score(y_test, y_pred):.6f} {recall_score(y_test, y_pred, average='macro'):.6f} {precision_score(y_test, y_pred, average='macro'):.6f}")
        
if __name__ == '__main__':
    main()