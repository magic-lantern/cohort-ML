import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
from sklearn.metrics import confusion_matrix, plot_roc_curve, roc_auc_score, roc_curve, auc, accuracy_score, balanced_accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.feature_selection import RFE, RFECV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

import timeit

import xgboost as xgb

# set standard random state for repeatability
my_random_state = 42

def fit_and_report(estimator=None, label='', datadict={}, features=[], ax=None):
    x_test = datadict['x_test']
    y_test = datadict['y_test']
    x_train = datadict['x_train']
    y_train = datadict['y_train']
    estimator.fit(x_train, y_train)

    print(label, '------------------------------------')
    # summarize the selection of the attributes
    if hasattr(estimator, 'feature_importances_'):
        importances = estimator.feature_importances_
        indices = np.argsort(importances)[::-1]
    elif hasattr(estimator, 'coef_'):
        importances = abs(estimator.coef_[0])
        indices = np.argsort(importances)[::-1]
    else:
        importances = []
        indices = []
    
    arr = []
    if len(importances) > 0:
        for f in range(x_train.shape[1]):
            print("%2d) %-*s %f" % (f + 1, 40, 
                                    features[indices[f]], 
                                    importances[indices[f]]))
            arr.append([features[indices[f]], importances[indices[f]]])

    y_pred = estimator.predict(x_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print(confmat)
    print('Balanced Accuracy:', balanced_accuracy_score(y_test, y_pred))
    print('Precision:', precision_score(y_test, y_pred))
    print('Recall:', recall_score(y_test, y_pred))
    y_pred = estimator.predict_proba(x_test)[:, 1]
    print('ROC_AUC_SCORE: ', roc_auc_score(y_true=y_test, y_score=y_pred))
    plot_roc_curve(estimator, x_test, y_test, ax=ax)
    print('------------------------------------')
    return pd.DataFrame(columns=[label + '_feature', label + '_importance'], data=arr)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.e4922e37-cdb5-4f8b-ae2a-bb41c11dcada"),
    data_encoded_and_outcomes=Input(rid="ri.foundry.main.dataset.32069249-a675-4faf-9d3c-a68ff0670c07"),
    data_scaled_and_outcomes=Input(rid="ri.foundry.main.dataset.b474df3d-909d-4a81-9e38-515e22b9cff3"),
    inpatient_encoded_w_imputation=Input(rid="ri.foundry.main.dataset.d3578a81-014a-49a6-9887-53d296155bdd"),
    inpatient_scaled_w_imputation=Input(rid="ri.foundry.main.dataset.f410db35-59e0-4b82-8fa8-d6dc6a61c9f2"),
    outcomes=Input(rid="ri.foundry.main.dataset.3d9b1654-3923-484f-8db5-6b38b56e290c")
)
def generate_models_and_summary_info(data_scaled_and_outcomes, inpatient_scaled_w_imputation, data_encoded_and_outcomes, outcomes, inpatient_encoded_w_imputation):
    # this set is for tree based methods that do not need/require scaling of the input data
    # categoricals have been one-hot encoded, imputation done, but no scaling
    data_and_outcomes = data_encoded_and_outcomes
    my_data_enc = data_and_outcomes.select(inpatient_encoded_w_imputation.columns).toPandas()
    my_data_enc = my_data_enc.drop(columns='visit_occurrence_id')
    my_outcomes = data_and_outcomes.select(outcomes.columns).toPandas()
    y = my_outcomes.bad_outcome
    x_train_enc, x_test_enc, y_train, y_test = train_test_split(my_data_enc, y, test_size=0.3, random_state=my_random_state, stratify=y)
    data_enc = {'x_train': x_train_enc, 'x_test': x_test_enc, 'y_train': y_train, 'y_test': y_test}

    # this version has alredy had StandardScaler applied to the data
    # after one-hot encoding, imputation
    data_and_outcomes_std = data_scaled_and_outcomes
    my_data_std = data_and_outcomes.select(inpatient_scaled_w_imputation.columns).toPandas()
    my_data_std = my_data_std.drop(columns='visit_occurrence_id')
    # y is just a binary outcome, so overwriting from previous train_test_split is ok
    x_train_std, x_test_std, y_train, y_test = train_test_split(my_data_std, y, test_size=0.3, random_state=my_random_state, stratify=y)
    data_std = {'x_train': x_train_std, 'x_test': x_test_std, 'y_train': y_train, 'y_test': y_test}

    # Axis to combine plots
    ax = plt.gca()

    #########################
    # Random Forest
    # best features from grid search: {'criterion': 'gini', 'max_features': 'sqrt', 'min_samples_split': 5, 'n_estimators': 500}
    #                                 {'criterion': 'gini', 'max_features': 'sqrt', 'min_samples_split': 5, 'n_estimators': 250}
    # parameters = {
    #    'n_estimators':[100,250,500,750,1000,1250],
    #    'criterion': ['gini', 'entropy'],
    #    'min_samples_split': [2, 5, 10, 20],
    #    'max_features' : ['sqrt', 'log2']
    # }
    #########################
    start = timeit.default_timer()
    rf = RandomForestClassifier(n_estimators=250,
                                min_samples_split=5,
                                random_state=my_random_state,
                                max_features='sqrt',
                                criterion='gini')
    rf_features = fit_and_report(estimator=rf, label='RandomForest', datadict=data_enc, features=my_data_enc.columns, ax=ax)
    stop = timeit.default_timer()
    print('Time: ', stop - start)  

    #########################
    # XGBoost 
    # best features from grid search {'booster': 'gbtree', 'learning_rate': 0.01, 'n_estimators': 1250}
    #                                {'booster': 'gbtree', 'learning_rate': 0.01, 'n_estimators': 1000}
    # parameters = {
    #    'n_estimators': [50,100,250,500,750,1000,1250],
    #    'learning_rate': [0.005, 0.01, 0.03, 0.06, 1],
    #    'booster': ['gbtree', 'gblinear', 'dart']
    # }
    #########################
    start = timeit.default_timer()
    xgb_model = xgb.XGBClassifier(n_jobs=4, # parallelization
                                  use_label_encoder=False,
                                  random_state=my_random_state,
                                  booster='gbtree',
                                  learning_rate=0.01,
                                  n_estimators=1250)
    xgb_features = fit_and_report(estimator=xgb_model, label='XGBoost', datadict=data_enc, features=my_data_enc.columns, ax=ax)
    stop = timeit.default_timer()
    print('Time: ', stop - start) 

    #########################
    # Logistic Regression
    # best featrues from grid search {'C': 1.0, 'penalty': 'l1', 'solver': 'liblinear'}
    # parameters = {
    #    'penalty': ['none', 'l1', 'l2', 'elasticnet'],
    #    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    #    'C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    # }
    #########################
    start = timeit.default_timer()
    lr = LogisticRegression(penalty='l1',
                            C=1.0,
                            random_state=my_random_state,
                            solver='liblinear',
                            max_iter=10000)
    lr_features = fit_and_report(estimator=lr, label='LogisticRegression', datadict=data_std, features=my_data_std.columns, ax=ax)
    stop = timeit.default_timer()
    print('Time: ', stop - start)

    #########################
    # Support Vector Machine
    # best featrues from grid search ?
    # parameters = {
    #    'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
    #    'gamma': ['scale', 'auto', 0.1, 0.2, 1.0, 10.0],
    #    'C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    # }
    #########################
    start = timeit.default_timer()
    svm = SVC(random_state=my_random_state,
              probability=True,
              cache_size=1600,
              kernel='rbf',
              gamma='scale',
              C=1.0)
    svm_features = fit_and_report(estimator=svm, label='SVM', datadict=data_std, features=my_data_std.columns, ax=ax)
    stop = timeit.default_timer()
    print('Time: ', stop - start)

    plt.show()

    return pd.concat([rf_features, xgb_features, lr_features, svm_features], axis=1)

@transform_pandas(
    Output(rid="ri.vector.main.execute.874e0c00-eb1e-49df-a1e7-31693b62f3b5"),
    data_encoded_and_outcomes=Input(rid="ri.foundry.main.dataset.32069249-a675-4faf-9d3c-a68ff0670c07"),
    data_scaled_and_outcomes=Input(rid="ri.foundry.main.dataset.b474df3d-909d-4a81-9e38-515e22b9cff3"),
    inpatient_encoded_w_imputation=Input(rid="ri.foundry.main.dataset.d3578a81-014a-49a6-9887-53d296155bdd"),
    inpatient_scaled_w_imputation=Input(rid="ri.foundry.main.dataset.f410db35-59e0-4b82-8fa8-d6dc6a61c9f2"),
    outcomes=Input(rid="ri.foundry.main.dataset.3d9b1654-3923-484f-8db5-6b38b56e290c")
)
def testing(data_scaled_and_outcomes, inpatient_scaled_w_imputation, data_encoded_and_outcomes, outcomes, inpatient_encoded_w_imputation):
    # this set is for tree based methods that do not need/require scaling of the input data
    # categoricals have been one-hot encoded, imputation done, but no scaling
    data_and_outcomes = data_encoded_and_outcomes
    my_data_enc = data_and_outcomes.select(inpatient_encoded_w_imputation.columns).toPandas()
    my_data_enc = my_data_enc.drop(columns='visit_occurrence_id')
    my_outcomes = data_and_outcomes.select(outcomes.columns).toPandas()
    y = my_outcomes.bad_outcome
    x_train_enc, x_test_enc, y_train, y_test = train_test_split(my_data_enc, y, test_size=0.3, random_state=my_random_state, stratify=y)
    data_enc = {'x_train': x_train_enc, 'x_test': x_test_enc, 'y_train': y_train, 'y_test': y_test}

    # this version has alredy had StandardScaler applied to the data
    # after one-hot encoding, imputation
    data_and_outcomes_std = data_scaled_and_outcomes
    my_data_std = data_and_outcomes.select(inpatient_scaled_w_imputation.columns).toPandas()
    my_data_std = my_data_std.drop(columns='visit_occurrence_id')
    # y is just a binary outcome, so overwriting from previous train_test_split is ok
    x_train_std, x_test_std, y_train, y_test = train_test_split(my_data_std, y, test_size=0.3, random_state=my_random_state, stratify=y)
    data_std = {'x_train': x_train_std, 'x_test': x_test_std, 'y_train': y_train, 'y_test': y_test}

    # Axis to combine plots
    ax = plt.gca()

    #########################
    # Random Forest
    # best features from grid search: {'criterion': 'gini', 'max_features': 'sqrt', 'min_samples_split': 5, 'n_estimators': 500}
    #                                 {'criterion': 'gini', 'max_features': 'sqrt', 'min_samples_split': 5, 'n_estimators': 250}
    # parameters = {
    #    'n_estimators':[100,250,500,750,1000,1250],
    #    'criterion': ['gini', 'entropy'],
    #    'min_samples_split': [2, 5, 10, 20],
    #    'max_features' : ['sqrt', 'log2']
    # }
    #########################
    start = timeit.default_timer()
    rf = RandomForestClassifier(n_estimators=250,
                                min_samples_split=5,
                                random_state=my_random_state,
                                max_features='sqrt',
                                criterion='gini')
    #rf_features = fit_and_report(estimator=rf, label='RandomForest', datadict=data_enc, features=my_data_enc.columns, ax=ax)
    stop = timeit.default_timer()
    print('Time: ', stop - start)  

    #########################
    # XGBoost 
    # best features from grid search {'booster': 'gbtree', 'learning_rate': 0.01, 'n_estimators': 1250}
    #                                {'booster': 'gbtree', 'learning_rate': 0.01, 'n_estimators': 1000}
    # parameters = {
    #    'n_estimators': [50,100,250,500,750,1000,1250],
    #    'learning_rate': [0.005, 0.01, 0.03, 0.06, 1],
    #    'booster': ['gbtree', 'gblinear', 'dart']
    # }
    #########################
    start = timeit.default_timer()
    xgb_model = xgb.XGBClassifier(n_jobs=4, # parallelization
                                  use_label_encoder=False,
                                  random_state=my_random_state,
                                  booster='gbtree',
                                  learning_rate=0.01,
                                  n_estimators=1000,
                                  objective = 'binary:logistic')
    xgb_features = fit_and_report(estimator=xgb_model, label='XGBoost', datadict=data_enc, features=my_data_enc.columns, ax=ax)
    stop = timeit.default_timer()
    print('Time: ', stop - start) 

    #########################
    # Logistic Regression
    # best featrues from grid search {'C': 1.0, 'penalty': 'l1', 'solver': 'liblinear'}
    # parameters = {
    #    'penalty': ['none', 'l1', 'l2', 'elasticnet'],
    #    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    #    'C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    # }
    #########################
    start = timeit.default_timer()
    lr = LogisticRegression(penalty='l1',
                            C=1.0,
                            random_state=my_random_state,
                            solver='liblinear',
                            max_iter=10000)
    #lr_features = fit_and_report(estimator=lr, label='LogisticRegression', datadict=data_std, features=my_data_std.columns, ax=ax)
    stop = timeit.default_timer()
    print('Time: ', stop - start)

    #########################
    # Support Vector Machine
    # best featrues from grid search 
    # parameters = {
    #    'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
    #    'gamma': ['scale', 'auto', 0.1, 0.2, 1.0, 10.0],
    #    'C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    # }
    #########################
    start = timeit.default_timer()
    svm = SVC(random_state=my_random_state,
              probability=True,
              cache_size=1600,
              kernel='rbf',
              gamma='auto',
              C=1.0)
    #svm_features = fit_and_report(estimator=svm, label='SVM', datadict=data_std, features=my_data_std.columns, ax=ax)
    stop = timeit.default_timer()
    print('Time: ', stop - start)

    plt.show()

    return pd.concat([rf_features, xgb_features, lr_features, svm_features], axis=1)

