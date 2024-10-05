# https://stackoverflow.com/questions/46498455/categorical-features-correlation/46498792#46498792
# https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import scipy.stats as ss

"""
def choose_numerical_features(ds, possible_features):
    correlated_features = []
    x = list(ds["label"])
    for feature in possible_features:
        if abs(np.corrcoef(x, list(ds[feature]))[0][1]) > 0.3:
            correlated_features.append(feature)
    return correlated_features
"""

def correlation_ratio(categories, measurements):
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)))
    denominator = np.sum(np.power(np.subtract(measurements, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator/denominator)
    return eta


def choose_numerical_features(ds, possible_features):
    correlated_features = []
    x = ds["label"]

    for feature in possible_features:
        if correlation_ratio(x, ds[feature]) > 0.5:
            correlated_features.append(feature)

    return correlated_features


def cramers_v(cm):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(cm)[0]
    n = cm.sum()
    phi2 = chi2 / n
    r, k = cm.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


def choose_categorical_features(ds, possible_features):
    correlated_features = []
    x = list(ds["label"])
    for feature in possible_features:
        cm = pd.crosstab(x, ds[feature])
        if cramers_v(cm.values) > 0.5:
            correlated_features.append(feature)
    return correlated_features


def data_preprocessing(ds_train, ds_test):
    # numerical columns
    num_cols = [col for col in ds_train.columns if ds_train[col].dtype in ["int64", "float64"]
                and col not in ["label", "sport", "dsport", "is_sm_ips_ports", "is_ftp_login", "swin", "dwin"]]
    # categorical columns to o-h encoding
    cat_cols = [col for col in ds_train.columns if (ds_train[col].dtype == "object" or col not in num_cols)
                and col != "label" and ds_train[col].nunique() < 15]

    # choosing features for training (correlated numerical columns)
    num_features_for_training = choose_numerical_features(ds_train, num_cols)
    # choosing features for training (correlated categorical columns)
    cat_features_for_training = choose_categorical_features(ds_train, cat_cols)

    # removing categorical columns with too many unique values; dividing datasets
    X_train_full = ds_train[num_features_for_training + cat_features_for_training].copy()
    X_test = ds_test[num_features_for_training + cat_features_for_training].copy()
    y_train_full = pd.DataFrame(ds_train["label"].copy())
    y_test = pd.DataFrame(ds_test["label"].copy())


    num_transformer = SimpleImputer(strategy="most_frequent")
    cat_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent",)),
                                      ("onehot", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_features_for_training),
                                                   ("cat", cat_transformer, cat_features_for_training)], sparse_threshold=0)

    # preprocessing
    pre_X_train_full = pd.DataFrame(preprocessor.fit_transform(X_train_full), columns=preprocessor.get_feature_names_out())
    pre_X_test = pd.DataFrame(preprocessor.fit_transform(X_test), columns=preprocessor.get_feature_names_out())


    #dividing into training and validation datasets
    X_train, X_valid, y_train, y_valid = train_test_split(pre_X_train_full, y_train_full,
                                                          train_size=0.8, test_size=0.2, random_state=0)

    return X_train, y_train, X_valid, y_valid, pre_X_test, y_test