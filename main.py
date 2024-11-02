# mrwellsdavid/unsw-nb15
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from preprocessing import data_preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


dataset_train_full = pd.read_csv("../unsw-nb15/UNSW_NB15_training-set.csv")
dataset_test_full = pd.read_csv("../unsw-nb15/UNSW_NB15_testing-set.csv")

dataset_train = dataset_train_full.drop(["attack_cat", "id"], axis=1)
dataset_test = dataset_test_full.drop(["attack_cat", "id"], axis=1)

type_change = ["is_sm_ips_ports", "is_ftp_login", "swin", "dwin"]
dataset_train[type_change] = dataset_train[type_change].astype(str)
dataset_test[type_change] = dataset_test[type_change].astype(str)

#dataset_train["label"] = 1 - dataset_train["label"]
#dataset_test["label"] = 1 - dataset_train["label"]

X_train, y_train, X_valid, y_valid, X_test, y_test = data_preprocessing(dataset_train, dataset_test)


### isolation forest

model_IF = IsolationForest(n_estimators=200, random_state=42, warm_start=True)
#model_IF.fit(X_train, y_train)
model_IF.fit(X_train)

validation_IF = model_IF.predict(X_valid)
validation_IF[validation_IF == -1] = 0
print(type(validation_IF))
cm_valid_IF = confusion_matrix(y_valid, validation_IF)
print(cm_valid_IF)
print("% of corrected predictions [IF]: ", (cm_valid_IF[0, 0]+cm_valid_IF[1, 1])/np.matrix(cm_valid_IF).sum())

### k-means

model_kmeans = KMeans(n_clusters=2, init="k-means++")
model_kmeans.fit(X_train, y_train)

validation_kmeans = model_kmeans.predict(X_valid)
cm_valid_kmeans = confusion_matrix(y_valid, validation_kmeans)
print(cm_valid_kmeans)
print("% of corrected predictions [kMeans]: ", (cm_valid_kmeans[0, 0]+cm_valid_kmeans[1, 1]) / np.matrix(cm_valid_kmeans).sum())


### LOF

model_LOF = LocalOutlierFactor()
model_LOF.fit(X_train, y_train)

validation_LOF = model_LOF.fit_predict(X_valid)
validation_LOF[validation_LOF == -1] = 0
cm_valid_LOF = confusion_matrix(y_valid, validation_LOF)
print(cm_valid_LOF)
print("% of corrected predictions [LOF]: ", (cm_valid_LOF[0, 0]+cm_valid_LOF[1, 1]) / np.matrix(cm_valid_LOF).sum())


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
print("% of corrected predictions: ", (cm_valid_mixed[0, 0]+cm_valid_mixed[1, 1]) / np.matrix(cm_valid_mixed).sum())


### logistic regression

model_LR = LogisticRegression(random_state=0)
model_LR.fit(X_train, y_train)

validation_LR = model_LR.predict(X_valid)
cm_valid_LR = confusion_matrix(y_valid, validation_LR)
print(cm_valid_LR)
print("% of corrected predictions [LR]: ", (cm_valid_LR[0, 0]+cm_valid_LR[1, 1]) / np.matrix(cm_valid_LR).sum())


### k-nn
