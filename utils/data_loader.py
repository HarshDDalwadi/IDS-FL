import pandas as pd
from sklearn.preprocessing import Normalizer
import numpy as np

def get_data():
    traindata = pd.read_csv('data/kddtrain.csv', header=None)
    testdata = pd.read_csv('data/kddtest.csv', header=None)
    X_train = traindata.iloc[:,1:42]
    Y_train = traindata.iloc[:,0]
    Y_test = testdata.iloc[:,0]
    X_test = testdata.iloc[:,1:42]
    
    scaler = Normalizer().fit(X_train)
    X_train = scaler.transform(X_train)

    scaler = Normalizer().fit(X_test)
    X_test = scaler.transform(X_test)


    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    return X_train, Y_train, X_test, Y_test
