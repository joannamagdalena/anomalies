# mrwellsdavid/unsw-nb15
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


def data_preprocessing(ds_train, ds_test):
    # numerical columns
    num_cols = [col for col in ds_train.columns if ds_train[col].dtype in ["int64", "float64"]]
    # categorical columns
    cat_cols = [col for col in ds_train.columns if ds_train[col].dtype == "object"
                and ds_train[col].nunique() < 15]

    pre_ds_train = ds_train[num_cols + cat_cols].copy()
    pre_ds_test = ds_test[num_cols + cat_cols].copy()

    num_transformer = SimpleImputer(strategy="most_frequent")
    cat_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                                      ("onehot", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_cols),
                                                   ("cat", cat_transformer, cat_cols)])

    preprocessor.fit_transform(pre_ds_train)
    preprocessor.fit(pre_ds_test)

    y_train = pre_ds_train["label"]
    y_test = pre_ds_test["label"]
    X_train = pre_ds_train.drop("label", axis=1)
    X_test = pre_ds_test.drop("label", axis=1)

    return X_train, y_train, X_test, y_test


dataset_train_full = pd.read_csv("../unsw-nb15/UNSW_NB15_training-set.csv")
dataset_test_full = pd.read_csv("../unsw-nb15/UNSW_NB15_testing-set.csv")

dataset_train = dataset_train_full.drop("attack_cat", axis=1)
dataset_test = dataset_test_full.drop("attack_cat", axis=1)

X_train, y_train, X_test, y_test = data_preprocessing(dataset_train, dataset_test)

