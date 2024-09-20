# mrwellsdavid/unsw-nb15
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix


def data_preprocessing(ds_train, ds_test):
    # numerical columns
    num_cols = [col for col in ds_train.columns if ds_train[col].dtype in ["int64", "float64"]
                and col != "label"]
    # categorical columns to o-h encoding
    cat_cols = [col for col in ds_train.columns if ds_train[col].dtype == "object"
                and ds_train[col].nunique() < 15]

    num_transformer = SimpleImputer(strategy="most_frequent")
    cat_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                                      ("onehot", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_cols),
                                                   ("cat", cat_transformer, cat_cols)])

    # removing categorical columns with too many unique values; dividing datasets
    X_train_full = ds_train[num_cols + cat_cols].copy()
    X_test = ds_test[num_cols + cat_cols].copy()
    y_train_full = pd.DataFrame(ds_train["label"].copy())
    y_test = pd.DataFrame(ds_test["label"].copy())

    # preprocessing
    pre_X_train_full = pd.DataFrame(preprocessor.fit_transform(X_train_full), columns=preprocessor.get_feature_names_out())
    pre_X_test = pd.DataFrame(preprocessor.fit_transform(X_test), columns=preprocessor.get_feature_names_out())

    X_train, X_valid, y_train, y_valid = train_test_split(pre_X_train_full, y_train_full,
                                                          train_size=0.8, test_size=0.2, random_state=1)

    return X_train, y_train, X_valid, y_valid, pre_X_test, y_test


dataset_train_full = pd.read_csv("../unsw-nb15/UNSW_NB15_training-set.csv")
dataset_test_full = pd.read_csv("../unsw-nb15/UNSW_NB15_testing-set.csv")

dataset_train = dataset_train_full.drop("attack_cat", axis=1)
dataset_test = dataset_test_full.drop("attack_cat", axis=1)

X_train, y_train, X_valid, y_valid, X_test, y_test = data_preprocessing(dataset_train, dataset_test)


### isolation forest

model_IF = IsolationForest(random_state=42)
model_IF.fit(X_train, y_train)

validation_IF = model_IF.predict(X_valid)
validation_IF[validation_IF == -1] = 0
cm_valid = confusion_matrix(y_valid, validation_IF)
print(cm_valid)
print("% of corrected predictions: ", (cm_valid[0,0]+cm_valid[1,1])/np.matrix(cm_valid).sum())


