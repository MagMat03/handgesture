import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    RANDOM_STATE,
    RF_N_ESTIMATORS,
    MLP_HIDDEN_LAYERS,
    MLP_MAX_ITER,
    SVM_KERNEL,
    SVM_C
)
def split_by_person(X, y, persons, test_person):

    test_mask = persons == test_person
    train_mask = ~test_mask
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    return X_train, X_test, y_train, y_test

def create_models():

    rf = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS,random_state=RANDOM_STATE)
    svm = Pipeline([("scaler", StandardScaler()),("classifier", SVC(kernel=SVM_KERNEL,C=SVM_C))])
    mlp = Pipeline([("scaler", StandardScaler()),("classifier", MLPClassifier(hidden_layer_sizes=MLP_HIDDEN_LAYERS,max_iter=MLP_MAX_ITER,random_state=RANDOM_STATE))])

    return rf, svm, mlp

def train_models(X_train, y_train):

    rf, svm, mlp = create_models()

    rf.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    mlp.fit(X_train, y_train)

    return rf, svm, mlp

