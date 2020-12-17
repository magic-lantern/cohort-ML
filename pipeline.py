import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
from functools import reduce
from sklearn.metrics import confusion_matrix, plot_roc_curve, roc_auc_score, roc_curve, auc, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.feature_selection import RFE, RFECV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

import timeit

import xgboost as xgb

# set standard random state for repeatability
my_random_state = 42

def fit_and_report(estimator=None, label='', datadict={}, features=[], ax=None, skip_predict_proba=False):
    x_test = datadict.get('x_test')
    y_test = datadict.get('y_test')
    x_train = datadict.get('x_train')
    y_train = datadict.get('y_train')
    mar_x_test = datadict.get('mar_x_test')
    mar_y_test = datadict.get('mar_y_test')
    jun_x_test = datadict.get('jun_x_test')
    jun_y_test = datadict.get('jun_y_test')

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
            arr.append([features[indices[f]], importances[indices[f]], f])
    print(arr)
    arr = arr + model_metrics(estimator, x_test, y_test, skip_predict_proba=skip_predict_proba)
    if (mar_x_test is not None):
        arr = arr + model_metrics(estimator, mar_x_test, mar_y_test, skip_predict_proba=skip_predict_proba, label='_Mar_to_May')
    if (jun_x_test is not None):
        arr = arr + model_metrics(estimator, jun_x_test, jun_y_test, skip_predict_proba=skip_predict_proba, label='_Jun_to_Oct')

    if ax is not None:
        plot_roc_curve(estimator, x_test, y_test, name=label, ax=ax)
    print('------------------------------------')
    return pd.DataFrame(columns=['variable', label, label + '_feature_rank'], data=arr).sort_values('variable')

# pull this out to separate function to reduce code in fit_and_report
def model_metrics(estimator=None, x_test=None, y_test=None, skip_predict_proba=False, label=''):
    arr = []

    if label is not '':
        print('Model metrics for ', label)
    
    y_pred = estimator.predict(x_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    #arr.append(['Confusion Matrix' + label, confmat.tostring()])
    print(confmat)
    arr.append(['z_Balanced Accuracy' + label, balanced_accuracy_score(y_test, y_pred), np.NaN])
    print('Balanced Accuracy:', balanced_accuracy_score(y_test, y_pred))
    arr.append(['z_Precision' + label, precision_score(y_test, y_pred), np.NaN])
    print('Precision:', precision_score(y_test, y_pred))
    arr.append(['z_Recall' + label, recall_score(y_test, y_pred), np.NaN])
    print('Recall:', recall_score(y_test, y_pred))
    arr.append(['z_F1' + label, f1_score(y_test, y_pred), np.NaN])
    print('F1:', f1_score(y_test, y_pred))
    if (not skip_predict_proba):
        y_pred = estimator.predict_proba(x_test)[:, 1]
        arr.append(['z_ROC_AUC_SCORE' + label, roc_auc_score(y_true=y_test, y_score=y_pred), np.NaN])
        print('ROC_AUC_SCORE: ', roc_auc_score(y_true=y_test, y_score=y_pred))
    return arr

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.e4922e37-cdb5-4f8b-ae2a-bb41c11dcada"),
    data_encoded_and_outcomes=Input(rid="ri.foundry.main.dataset.32069249-a675-4faf-9d3c-a68ff0670c07"),
    data_scaled_and_outcomes=Input(rid="ri.foundry.main.dataset.b474df3d-909d-4a81-9e38-515e22b9cff3"),
    inpatient_encoded_w_imputation=Input(rid="ri.foundry.main.dataset.d3578a81-014a-49a6-9887-53d296155bdd"),
    inpatient_scaled_w_imputation=Input(rid="ri.foundry.main.dataset.f410db35-59e0-4b82-8fa8-d6dc6a61c9f2"),
    jun_to_oct_encoded_and_outcomes=Input(rid="ri.foundry.main.dataset.b260be3e-e48d-4428-9a44-e4ceb10113e5"),
    jun_to_oct_scaled_and_outcomes=Input(rid="ri.foundry.main.dataset.bab694df-4318-4c0e-aa36-b0f4296c6360"),
    mar_to_may_encoded_and_outcomes=Input(rid="ri.foundry.main.dataset.fd6475f7-d8dc-4601-a3ce-0e7e3d166da3"),
    mar_to_may_scaled_and_outcomes=Input(rid="ri.foundry.main.dataset.c0fd81e6-dc02-45b9-93fe-b0047394e4f8"),
    outcomes=Input(rid="ri.foundry.main.dataset.3d9b1654-3923-484f-8db5-6b38b56e290c")
)
def generate_models_and_summary_info(data_scaled_and_outcomes, inpatient_scaled_w_imputation, data_encoded_and_outcomes, outcomes, inpatient_encoded_w_imputation, mar_to_may_scaled_and_outcomes, jun_to_oct_scaled_and_outcomes, jun_to_oct_encoded_and_outcomes, mar_to_may_encoded_and_outcomes):
    # this set is for tree based methods that do not need/require scaling of the input data
    # categoricals have been one-hot encoded, imputation done, but no scaling
    data_and_outcomes = data_encoded_and_outcomes.toPandas()
    my_data_enc = data_and_outcomes[inpatient_encoded_w_imputation.columns]
    my_outcomes = data_and_outcomes[outcomes.columns]
    # this version has alredy had StandardScaler applied to the data
    # after one-hot encoding, imputation
    data_and_outcomes_std = data_scaled_and_outcomes.toPandas()
    my_data_std = data_and_outcomes_std[inpatient_scaled_w_imputation.columns]
    # outcome
    y = my_outcomes.bad_outcome
    # split dataset
    x_train_enc, x_test_enc, y_train_enc, y_test_enc = train_test_split(my_data_enc, y, test_size=0.3, random_state=my_random_state, stratify=y)

    # setup the train/test split with the standardized version of the dataset
    x_train_std = my_data_std[my_data_std.visit_occurrence_id.isin(x_train_enc.visit_occurrence_id)]
    y_train_std = data_and_outcomes_std[data_and_outcomes_std.visit_occurrence_id.isin(x_train_enc.visit_occurrence_id)].bad_outcome
    x_test_std = my_data_std[my_data_std.visit_occurrence_id.isin(x_test_enc.visit_occurrence_id)]
    y_test_std = data_and_outcomes_std[data_and_outcomes_std.visit_occurrence_id.isin(x_test_enc.visit_occurrence_id)].bad_outcome

    # figure out what from seasonal dataset are part of test
    mar_x_test_enc = mar_to_may_encoded_and_outcomes.toPandas()
    mar_x_test_enc = mar_x_test_enc[mar_x_test_enc.visit_occurrence_id.isin(x_test_enc.visit_occurrence_id)]
    jun_x_test_enc = jun_to_oct_encoded_and_outcomes.toPandas()
    jun_x_test_enc = jun_x_test_enc[jun_x_test_enc.visit_occurrence_id.isin(x_test_enc.visit_occurrence_id)]

    # drop visit_occurrence_id from x_ datasets
    x_train_enc = x_train_enc.drop(columns='visit_occurrence_id')
    x_test_enc = x_test_enc.drop(columns='visit_occurrence_id')
    mar_y_test_enc = mar_x_test_enc.bad_outcome
    mar_x_test_enc = mar_x_test_enc[x_test_enc.columns]
    jun_y_test_enc = jun_x_test_enc.bad_outcome
    jun_x_test_enc = jun_x_test_enc[x_test_enc.columns]

    data_enc = {'x_train': x_train_enc,
                'x_test': x_test_enc,
                'y_train': y_train_enc,
                'y_test': y_test_enc,
                'mar_x_test' : mar_x_test_enc,
                'mar_y_test': mar_y_test_enc,
                'jun_x_test' : jun_x_test_enc,
                'jun_y_test': jun_y_test_enc}

    # figure out what from seasonal dataset are part of test data that has been standardized
    mar_x_test_std = mar_to_may_scaled_and_outcomes.toPandas()
    mar_x_test_std = mar_x_test_std[mar_x_test_std.visit_occurrence_id.isin(x_test_std.visit_occurrence_id)]
    jun_x_test_std = jun_to_oct_scaled_and_outcomes.toPandas()
    jun_x_test_std = jun_x_test_std[jun_x_test_std.visit_occurrence_id.isin(x_test_std.visit_occurrence_id)]

    # drop visit_occurrence_id from x_ datasets
    x_train_std = x_train_std.drop(columns='visit_occurrence_id')
    x_test_std = x_test_std.drop(columns='visit_occurrence_id')
    mar_y_test_std = mar_x_test_std.bad_outcome
    mar_x_test_std = mar_x_test_std[x_test_std.columns]
    jun_y_test_std = jun_x_test_std.bad_outcome
    jun_x_test_std = jun_x_test_std[x_test_std.columns]

    data_std = {'x_train': x_train_std,
                'x_test': x_test_std,
                'y_train': y_train_std,
                'y_test': y_test_std,
                'mar_x_test' : mar_x_test_std,
                'mar_y_test': mar_y_test_std,
                'jun_x_test' : jun_x_test_std,
                'jun_y_test': jun_y_test_std}

    # drop columns to match x_ arrays
    my_data_enc = my_data_enc.drop(columns='visit_occurrence_id')
    my_data_std = my_data_std.drop(columns='visit_occurrence_id')

    # Axis to combine plots
    ax = plt.gca()

    #########################
    # XGBoost 
    # best features from grid search {'booster': 'gbtree', 'learning_rate': 0.01, 'n_estimators': 1250}
    #                                {'booster': 'gbtree', 'learning_rate': 0.01, 'n_estimators': 1000}
    # parameters = {
    #    'n_estimators': [50,100,250,500,750,1000,1250],
    #    'learning_rate': [0.005, 0.01, 0.03, 0.06, 1],
    #    # after finding 0.01 was best, tried 'learning_rate': np.arange(0.001, 0.2 , 0.0025)
    #    'booster': ['gbtree', 'gblinear', 'dart']
    # }
    #########################
    start = timeit.default_timer()
    xgb_model = xgb.XGBClassifier(n_jobs=4, # parallelization
                                  use_label_encoder=False,
                                  random_state=my_random_state,
                                  booster='gbtree',
                                  learning_rate=0.0385,
                                  n_estimators=500)
    xgb_features = fit_and_report(estimator=xgb_model, label='XGBoost', datadict=data_enc, features=my_data_enc.columns, ax=ax)
    stop = timeit.default_timer()
    print('Time: ', stop - start)

    #########################
    # Random Forest
    # best features from grid search: {'criterion': 'entropy', 'max_features': 'sqrt', 'min_samples_split': 9, 'n_estimators': 750}
    # parameters = {
    #    'n_estimators':[100,250,500,750,1000,1250],
    #    'criterion': ['gini', 'entropy'],
    #    'min_samples_split': range(2, 21),
    #    'max_features' : ['sqrt', 'log2']
    # }
    #########################
    start = timeit.default_timer()
    rf = RandomForestClassifier(n_estimators=750,
                                min_samples_split=9,
                                random_state=my_random_state,
                                max_features='sqrt',
                                criterion='entropy')
    rf_features = fit_and_report(estimator=rf, label='RandomForest', datadict=data_enc, features=my_data_enc.columns, ax=ax)
    stop = timeit.default_timer()
    print('Time: ', stop - start)

    #########################
    # Logistic Regression
    # best features from grid search {'C': 0.25, 'penalty': 'l1', 'solver': 'liblinear'}
    # parameters = {
    #    'penalty': ['none', 'l1', 'l2', 'elasticnet'],
    #    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    #    'C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    # }
    #########################
    # LR penalty none: {penalty': 'none', 'solver': 'newton-cg'}
    # penalty none ignores C and l1 ratio params, so not much to config here
    start = timeit.default_timer()
    lr = LogisticRegression(penalty='none',
                            random_state=my_random_state,
                            solver='newton-cg',
                            max_iter=10000)
    lr_none_features = fit_and_report(estimator=lr, label='LogisticRegression_None', datadict=data_std, features=my_data_std.columns, ax=ax)
    stop = timeit.default_timer()
    print('Time: ', stop - start)

    start = timeit.default_timer()
    lr = LogisticRegression(penalty='l1',
                            C=0.25,
                            random_state=my_random_state,
                            solver='liblinear',
                            max_iter=10000)
    lr_l1_features = fit_and_report(estimator=lr, label='LogisticRegression_L1', datadict=data_std, features=my_data_std.columns, ax=ax)
    stop = timeit.default_timer()
    print('Time: ', stop - start)

    # LR L2: {'C': 0.25, 'penalty': 'l2', 'solver': 'liblinear'}
    start = timeit.default_timer()
    lr = LogisticRegression(penalty='l2',
                            random_state=my_random_state,
                            C=0.25,
                            solver='liblinear',
                            max_iter=10000)
    lr_l2_features = fit_and_report(estimator=lr, label='LogisticRegression_L2', datadict=data_std, features=my_data_std.columns, ax=ax)
    stop = timeit.default_timer()
    print('Time: ', stop - start)

    # LR elasticnet {'l1_ratio': 0.45, 'penalty': 'elasticnet', 'solver': 'saga'}
    #       'l1_ratio': [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0]
    start = timeit.default_timer()
    lr = LogisticRegression(penalty='elasticnet',
                            random_state=my_random_state,
                            l1_ratio=0.45,
                            solver='saga',
                            max_iter=10000)
    lr_elastic_features = fit_and_report(estimator=lr, label='LogisticRegression_Elasticnet', datadict=data_std, features=my_data_std.columns)
    stop = timeit.default_timer()
    print('Time: ', stop - start)

    #########################
    # Ridge Classification
    # best features from grid search {'alpha': 0.7, 'solver': 'sparse_cg'}
    # parameters = {
    #    'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
    #    'alpha': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    # }
    start = timeit.default_timer()
    rc = RidgeClassifier(random_state=my_random_state,
                         alpha=0.7,
                         solver='sparse_cg',
                         class_weight='balanced')
    rc_features = fit_and_report(estimator=rc, label='RidgeClassifier', datadict=data_std, features=my_data_std.columns, skip_predict_proba=True)
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

    dfs = [rf_features, xgb_features, lr_none_features, lr_l1_features, lr_l2_features, lr_elastic_features, rc_features, svm_features]
    df_combined = reduce(lambda left,right: pd.merge(left,right,on='variable',how='outer'), dfs)
    # any columns that are all null should be dropped
    df_combined = df_combined.dropna(axis='columns', how='all')
    print(df_combined.columns)
    print(df_combined.head())
    return df_combined

@transform_pandas(
    Output(rid="ri.vector.main.execute.737ddbae-e975-4dd0-9ad4-dba1c4186c78"),
    inpatient_ml_dataset=Input(rid="ri.foundry.main.dataset.07927bca-b175-4775-9c55-a371af481cc1")
)
def inpatient_encoded(inpatient_ml_dataset):
    inpatient_ml_dataset = inpatient_ml_dataset
    # get rid of ids, columns that are duplicates of other information,
    # or columns that are from the end of stay
    sdf = inpatient_ml_dataset
    sdf = sdf.drop('covid_status_name')
    sdf = sdf.drop('person_id')
    sdf = sdf.drop('visit_concept_id')
    sdf = sdf.drop('visit_concept_name')
    sdf = sdf.drop('in_death_table')
    sdf = sdf.drop('severity_type')
    sdf = sdf.drop('length_of_stay')
    sdf = sdf.drop('ecmo')
    sdf = sdf.drop('aki_in_hospital')
    sdf = sdf.drop('invasive_ventilation')
    sdf = sdf.drop('testcount')
    sdf = sdf.drop('bad_outcome')
    # before this have filtered to patients affected by COVID
    sdf = sdf.drop('positive_covid_test', 'negative_covid_test', 'suspected_covid')

    # these columns are 85% or greater NULL (insurance information)
    sdf = sdf.drop('miscellaneous_program', 'department_of_corrections', 'department_of_defense', 'other_government_federal_state_local_excluding_department_of_corrections', 'no_payment_from_an_organization_agency_program_private_payer_listed', 'medicaid', 'private_health_insurance', 'medicare', 'payer_no_matching_concept')
    # these ones are 100% present, but real values are rare
    sdf = sdf.drop('smoking_status') # only 0.2% have smoking
    sdf = sdf.drop('blood_type')     # only 9% have a value besides unknown
    
    df = sdf.toPandas()

    # actually decided that these columns are 'cheating'
    # fixing columns so they work with sklearn
    # df['visit_start'] = pd.to_datetime(df.visit_start_date).astype('int64')
    # df['visit_end'] = pd.to_datetime(df.visit_end_date).astype('int64')
    df = df.drop(columns=['visit_start_date', 'visit_end_date'])
    
    # remove site so analysis is more generalized across all COVID positive
    # df = pd.concat([df, pd.get_dummies(df.data_partner_id, prefix='site', drop_first=True)], axis=1)
    df = pd.concat([df.drop('gender_concept_name', axis=1), pd.get_dummies(df.gender_concept_name, prefix='gender', drop_first=True)], axis=1)
    df = pd.concat([df.drop('race', axis=1), pd.get_dummies(df.race, prefix='race', drop_first=True)], axis=1)
    df = pd.concat([df.drop('ethnicity', axis=1), pd.get_dummies(df.ethnicity, prefix='ethnicity', drop_first=True)], axis=1)
    #df = pd.concat([df.drop('smoking_status', axis=1), pd.get_dummies(df.smoking_status, prefix='smoking', drop_first=True)], axis=1)
    #df = pd.concat([df.drop('blood_type', axis=1), pd.get_dummies(df.blood_type, prefix='blood_type', drop_first=True)], axis=1)
    #df = pd.concat([df.drop('severity_type', axis=1), pd.get_dummies(df.severity_type, prefix='severity', drop_first=True)], axis=1)

    # these boolean coluumns aren't being treated as boolean
    charlson_cols = ['chf', 'cancer', 'dm', 'dmcx', 'dementia', 'hiv', 'livermild', 'liversevere', 'mi', 'mets', 'pud', 'pvd', 'paralysis', 'pulmonary', 'renal', 'rheumatic', 'stroke']
    df[charlson_cols] = df[charlson_cols].astype('bool')

    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('/', '_')
    df.columns = df.columns.str.lower()
    
    return df
    

@transform_pandas(
    Output(rid="ri.vector.main.execute.1ce10b4d-a713-4b8f-a623-8f7e6edac75f"),
    inpatient_encoded=Input(rid="ri.vector.main.execute.737ddbae-e975-4dd0-9ad4-dba1c4186c78")
)
def inpatient_encoded_w_imputation_1(inpatient_encoded):
    df = inpatient_encoded
    # remove data_partner_id as it was kept for viewing missing data by site. Not part of the rest of this analysis pipeline
    df = df.drop(columns='data_partner_id')

    df['bnp_pg_ml'] = df['bnp_pg_ml'].fillna(100)
    df['c-reactive_protein_crp_mg_l'] = df['c-reactive_protein_crp_mg_l'].fillna(10)
    df['erythrocyte_sed_rate_mm_hr'] = df['erythrocyte_sed_rate_mm_hr'].fillna(19)
    df['lactate_mm'] = df['lactate_mm'].fillna(13.5)
    df['nt_pro_bnp_pg_ml'] = df['nt_pro_bnp_pg_ml'].fillna(125)
    df['procalcitonin_ng_ml'] = df['procalcitonin_ng_ml'].fillna(0.02)
    df['troponin_all_types_ng_ml'] = df['troponin_all_types_ng_ml'].fillna(0.02)

    df.loc[(df.gender_male == True) & (df.ferritin_ng_ml.isna()), 'ferritin_ng_ml'] = 150
    df.loc[(df.gender_male == False) & (df.gender_other == False) & (df.ferritin_ng_ml.isna()), 'ferritin_ng_ml'] = 75
    
    # fill these with False - now dropped due to already dropping other insurance info
    # df['medicare'] = df['medicare'].fillna(False)
    # df['payer_no_matching_concept'] = df['payer_no_matching_concept'].fillna(False)

    # now fill the rest with the median
    df = df.fillna(df.median())

    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.9e3c22ec-1a47-4bfa-bace-028a54a1c685"),
    data_encoded_and_outcomes=Input(rid="ri.foundry.main.dataset.32069249-a675-4faf-9d3c-a68ff0670c07"),
    data_scaled_and_outcomes=Input(rid="ri.foundry.main.dataset.b474df3d-909d-4a81-9e38-515e22b9cff3"),
    inpatient_encoded_w_imputation=Input(rid="ri.foundry.main.dataset.d3578a81-014a-49a6-9887-53d296155bdd"),
    inpatient_scaled_w_imputation=Input(rid="ri.foundry.main.dataset.f410db35-59e0-4b82-8fa8-d6dc6a61c9f2"),
    jun_to_oct_encoded_and_outcomes=Input(rid="ri.foundry.main.dataset.b260be3e-e48d-4428-9a44-e4ceb10113e5"),
    jun_to_oct_scaled_and_outcomes=Input(rid="ri.foundry.main.dataset.bab694df-4318-4c0e-aa36-b0f4296c6360"),
    mar_to_may_encoded_and_outcomes=Input(rid="ri.foundry.main.dataset.fd6475f7-d8dc-4601-a3ce-0e7e3d166da3"),
    mar_to_may_scaled_and_outcomes=Input(rid="ri.foundry.main.dataset.c0fd81e6-dc02-45b9-93fe-b0047394e4f8"),
    outcomes=Input(rid="ri.foundry.main.dataset.3d9b1654-3923-484f-8db5-6b38b56e290c")
)
def model_compare(data_scaled_and_outcomes, inpatient_scaled_w_imputation, data_encoded_and_outcomes, outcomes, inpatient_encoded_w_imputation, mar_to_may_scaled_and_outcomes, jun_to_oct_scaled_and_outcomes, jun_to_oct_encoded_and_outcomes, mar_to_may_encoded_and_outcomes):
    # this set is for tree based methods that do not need/require scaling of the input data
    # categoricals have been one-hot encoded, imputation done, but no scaling
    data_and_outcomes = data_encoded_and_outcomes.toPandas()
    my_data_enc = data_and_outcomes[inpatient_encoded_w_imputation.columns]
    my_outcomes = data_and_outcomes[outcomes.columns]
    # this version has alredy had StandardScaler applied to the data
    # after one-hot encoding, imputation
    data_and_outcomes_std = data_scaled_and_outcomes.toPandas()
    my_data_std = data_and_outcomes_std[inpatient_scaled_w_imputation.columns]
    # outcome
    y = my_outcomes.bad_outcome
    # split dataset
    x_train_enc, x_test_enc, y_train_enc, y_test_enc = train_test_split(my_data_enc, y, test_size=0.3, random_state=my_random_state, stratify=y)

    # setup the train/test split with the standardized version of the dataset
    x_train_std = my_data_std[my_data_std.visit_occurrence_id.isin(x_train_enc.visit_occurrence_id)]
    y_train_std = data_and_outcomes_std[data_and_outcomes_std.visit_occurrence_id.isin(x_train_enc.visit_occurrence_id)].bad_outcome
    x_test_std = my_data_std[my_data_std.visit_occurrence_id.isin(x_test_enc.visit_occurrence_id)]
    y_test_std = data_and_outcomes_std[data_and_outcomes_std.visit_occurrence_id.isin(x_test_enc.visit_occurrence_id)].bad_outcome

    # figure out what from seasonal dataset are part of test
    mar_x_test_enc = mar_to_may_encoded_and_outcomes.toPandas()
    mar_x_test_enc = mar_x_test_enc[mar_x_test_enc.visit_occurrence_id.isin(x_test_enc.visit_occurrence_id)]
    jun_x_test_enc = jun_to_oct_encoded_and_outcomes.toPandas()
    jun_x_test_enc = jun_x_test_enc[jun_x_test_enc.visit_occurrence_id.isin(x_test_enc.visit_occurrence_id)]

    # drop visit_occurrence_id from x_ datasets
    x_train_enc = x_train_enc.drop(columns='visit_occurrence_id')
    x_test_enc = x_test_enc.drop(columns='visit_occurrence_id')
    mar_y_test_enc = mar_x_test_enc.bad_outcome
    mar_x_test_enc = mar_x_test_enc[x_test_enc.columns]
    jun_y_test_enc = jun_x_test_enc.bad_outcome
    jun_x_test_enc = jun_x_test_enc[x_test_enc.columns]

    data_enc = {'x_train': x_train_enc,
                'x_test': x_test_enc,
                'y_train': y_train_enc,
                'y_test': y_test_enc,
                'mar_x_test' : mar_x_test_enc,
                'mar_y_test': mar_y_test_enc,
                'jun_x_test' : jun_x_test_enc,
                'jun_y_test': jun_y_test_enc}

    # figure out what from seasonal dataset are part of test data that has been standardized
    mar_x_test_std = mar_to_may_scaled_and_outcomes.toPandas()
    mar_x_test_std = mar_x_test_std[mar_x_test_std.visit_occurrence_id.isin(x_test_std.visit_occurrence_id)]
    jun_x_test_std = jun_to_oct_scaled_and_outcomes.toPandas()
    jun_x_test_std = jun_x_test_std[jun_x_test_std.visit_occurrence_id.isin(x_test_std.visit_occurrence_id)]

    # drop visit_occurrence_id from x_ datasets
    x_train_std = x_train_std.drop(columns='visit_occurrence_id')
    x_test_std = x_test_std.drop(columns='visit_occurrence_id')
    mar_y_test_std = mar_x_test_std.bad_outcome
    mar_x_test_std = mar_x_test_std[x_test_std.columns]
    jun_y_test_std = jun_x_test_std.bad_outcome
    jun_x_test_std = jun_x_test_std[x_test_std.columns]

    data_std = {'x_train': x_train_std,
                'x_test': x_test_std,
                'y_train': y_train_std,
                'y_test': y_test_std,
                'mar_x_test' : mar_x_test_std,
                'mar_y_test': mar_y_test_std,
                'jun_x_test' : jun_x_test_std,
                'jun_y_test': jun_y_test_std}

    # drop columns to match x_ arrays
    my_data_enc = my_data_enc.drop(columns='visit_occurrence_id')
    my_data_std = my_data_std.drop(columns='visit_occurrence_id')

    # Axis to combine plots
    ax = plt.gca()

    start = timeit.default_timer()
    rc = RidgeClassifier(random_state=my_random_state,
                         alpha=0.7,
                         solver='sparse_cg',
                         class_weight='balanced')
    rc_features = fit_and_report(estimator=rc, label='RidgeClassifier', datadict=data_std, features=my_data_std.columns, skip_predict_proba=True, ax=ax)
    stop = timeit.default_timer()
    print('Time: ', stop - start)

    #{'alpha': 0.16899999999999987, 'solver': 'svd'}
    start = timeit.default_timer()
    rc = RidgeClassifier(random_state=my_random_state,
                         alpha=0.169,
                         solver='svd',
                         class_weight='balanced')
    rc_features = fit_and_report(estimator=rc, label='RidgeClassifier_new', datadict=data_std, features=my_data_std.columns, skip_predict_proba=True, ax=ax)
    stop = timeit.default_timer()
    print('Time: ', stop - start)

    plt.show()
    return

