import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, accuracy_score, confusion_matrix, plot_roc_curve, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.feature_selection import RFE, RFECV
from sklearn.pipeline import Pipeline

import timeit

import xgboost as xgb

# set standard random state for repeatability
my_random_state = 42

@transform_pandas(
    Output(rid="ri.vector.main.execute.8006baaa-9271-4b84-8282-bb82a1ce4b40"),
    data_scaled_and_outcomes=Input(rid="ri.foundry.main.dataset.b474df3d-909d-4a81-9e38-515e22b9cff3")
)
def unnamed(data_scaled_and_outcomes):
    

