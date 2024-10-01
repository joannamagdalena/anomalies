# mrwellsdavid/unsw-nb15
# https://stackoverflow.com/questions/46498455/categorical-features-correlation/46498792#46498792
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
import scipy.stats as ss


def choose_numerical_features(ds, possible_features):
    correlated_features = []
    x = list(ds["label"])
    for feature in possible_features:
        if abs(np.corrcoef(x, list(ds[feature]))[0][1]) > 0.3:
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
                and col != "label"]
    # categorical columns to o-h encoding
    cat_cols = [col for col in ds_train.columns if ds_train[col].dtype == "object"
                and ds_train[col].nunique() < 15]

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
    cat_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                                      ("onehot", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_features_for_training),
                                                   ("cat", cat_transformer, cat_features_for_training)])

    # preprocessing
    pre_X_train_full = pd.DataFrame(preprocessor.fit_transform(X_train_full), columns=preprocessor.get_feature_names_out())
    pre_X_test = pd.DataFrame(preprocessor.fit_transform(X_test), columns=preprocessor.get_feature_names_out())


    #dividing into training and validation datasets
    X_train, X_valid, y_train, y_valid = train_test_split(pre_X_train_full, y_train_full,
                                                          train_size=0.8, test_size=0.2, random_state=1)

    return X_train, y_train, X_valid, y_valid, pre_X_test, y_test


dataset_train_full = pd.read_csv("../unsw-nb15/UNSW_NB15_training-set.csv")
dataset_test_full = pd.read_csv("../unsw-nb15/UNSW_NB15_testing-set.csv")

dataset_train = dataset_train_full.drop(["attack_cat", "id"], axis=1)
dataset_test = dataset_test_full.drop(["attack_cat", "id"], axis=1)

X_train, y_train, X_valid, y_valid, X_test, y_test = data_preprocessing(dataset_train, dataset_test)


### isolation forest

model_IF = IsolationForest(n_estimators=200, random_state=42, warm_start=True)
model_IF.fit(X_train, y_train)

validation_IF = model_IF.predict(X_valid)
validation_IF[validation_IF == -1] = 0
print(type(validation_IF))
cm_valid_IF = confusion_matrix(y_valid, validation_IF)
print(cm_valid_IF)
print("% of corrected predictions: ", (cm_valid_IF[0, 0]+cm_valid_IF[1, 1])/np.matrix(cm_valid_IF).sum())

### k-means

model_kmeans = KMeans(n_clusters=2, init="k-means++")
model_kmeans.fit(X_train, y_train)

validation_kmeans = model_kmeans.predict(X_valid)
cm_valid_kmeans = confusion_matrix(y_valid, validation_kmeans)
print(cm_valid_kmeans)
print("% of corrected predictions: ", (cm_valid_kmeans[0, 0]+cm_valid_kmeans[1, 1])/np.matrix(cm_valid_kmeans).sum())


### LOF

model_LOF = LocalOutlierFactor()
model_LOF.fit(X_train, y_train)

validation_LOF = model_LOF.fit_predict(X_valid)
validation_LOF[validation_LOF == -1] = 0
cm_valid_LOF = confusion_matrix(y_valid, validation_LOF)
print(cm_valid_LOF)
print("% of corrected predictions: ", (cm_valid_LOF[0, 0]+cm_valid_LOF[1, 1])/np.matrix(cm_valid_LOF).sum())


### mixed

validation_mixed = []
for i in range(0, len(validation_IF)):
    if validation_IF[i] == 0 and validation_LOF[i] == 0:
        validation_mixed.append(0)
    elif validation_IF[i] == 1 and validation_kmeans[i] == 1:
        validation_mixed.append(1)
    else:
        validation_mixed.append(validation_IF[i])

validation_mixed = np.array(validation_mixed)
cm_valid_mixed = confusion_matrix(y_valid, validation_mixed)
print(cm_valid_mixed)
print("% of corrected predictions: ", (cm_valid_mixed[0, 0]+cm_valid_mixed[1, 1])/np.matrix(cm_valid_mixed).sum())