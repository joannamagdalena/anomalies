import loading_data
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from preprocessing import data_preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
#import eli5
#from eli5.sklearn import PermutationImportance
from sklearn.inspection import permutation_importance


dataset_train, dataset_test = loading_data.loading_data_from_csv()

#dataset_train["label"] = 1 - dataset_train["label"]
#dataset_test["label"] = 1 - dataset_train["label"]

X_train, y_train, X_valid, y_valid, X_test, y_test, X_full, y_full = data_preprocessing(dataset_train, dataset_test)


###################################################################################
### isolation forest

model_IF = IsolationForest(n_estimators=200, random_state=42, warm_start=True)
#model_IF.fit(X_train, y_train)
model_IF.fit(X_train)

validation_IF = model_IF.predict(X_valid)
validation_IF[validation_IF == -1] = 0
cm_valid_IF = confusion_matrix(y_valid, validation_IF)
print(cm_valid_IF)
print("% of corrected predictions [IF]: ", (cm_valid_IF[0, 0]+cm_valid_IF[1, 1])/np.matrix(cm_valid_IF).sum())

###################################################################################
### k-means

model_kmeans = KMeans(n_clusters=2, init="k-means++")
model_kmeans.fit(X_train, y_train)

validation_kmeans = model_kmeans.predict(X_valid)
cm_valid_kmeans = confusion_matrix(y_valid, validation_kmeans)
print(cm_valid_kmeans)
print("% of corrected predictions [kMeans]: ", (cm_valid_kmeans[0, 0]+cm_valid_kmeans[1, 1]) / np.matrix(cm_valid_kmeans).sum())

###################################################################################
### LOF

model_LOF = LocalOutlierFactor()
model_LOF.fit(X_train, y_train)

validation_LOF = model_LOF.fit_predict(X_valid)
validation_LOF[validation_LOF == -1] = 0
cm_valid_LOF = confusion_matrix(y_valid, validation_LOF)
print(cm_valid_LOF)
print("% of corrected predictions [LOF]: ", (cm_valid_LOF[0, 0]+cm_valid_LOF[1, 1]) / np.matrix(cm_valid_LOF).sum())


'''
###################################################################################
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
'''

###################################################################################
### logistic regression

model_LR = LogisticRegression(random_state=0)
model_LR.fit(X_train, y_train.values.ravel())

validation_LR = model_LR.predict(X_valid)
cm_valid_LR = confusion_matrix(y_valid, validation_LR)
print(cm_valid_LR)
print("% of corrected predictions [LR]: ", (cm_valid_LR[0, 0]+cm_valid_LR[1, 1]) / np.matrix(cm_valid_LR).sum())

# cross-val for logistic regression
s = -1 * cross_val_score(model_LR, X_full, y_full.values.ravel(), cv=5, scoring='neg_mean_absolute_error')
print("cross-validation for LR: ", s)
print("avg error: ", s.mean())

###################################################################################
### k-nn

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train.values.ravel())

validation_KNN = knn.predict(X_valid)
cm_valid_KNN = confusion_matrix(y_valid, validation_KNN)
print(cm_valid_KNN)
print("% of corrected predictions [KNN]: ", (cm_valid_KNN[0, 0]+cm_valid_KNN[1, 1]) / np.matrix(cm_valid_KNN).sum())

###################################################################################
### xbgregressor

xbg = XGBRegressor(n_estimators=500, early_stopping_rounds=5, learning_rate=0.1)
xbg.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

validation_xbg = xbg.predict(X_valid)
print("mean absolute error for XGBRegressor: ", mean_absolute_error(validation_xbg, y_valid))

# permutation importance for xbgregressor
#permutations = PermutationImportance(xbg, random_state=1).fit(X_train, y_train)
#print(eli5.format_as_text(eli5.show_weights(permutations, feature_names=X_valid.columns.tolist())))
permutations = permutation_importance(xbg, X_valid, y_valid, n_repeats=30, random_state=0, scoring="neg_mean_absolute_error")
for i in permutations.importances_mean.argsort()[::-1]:
    print(f"{X_valid.columns.tolist()[i]:<8}", " ",
          f"{permutations.importances_mean[i]:.3f}", " ",
          f" +/- {permutations.importances_std[i]:.3f}")

# comparison
X_train_2 = X_train[["num__sttl","cat__state_FIN","cat__state_REQ"]].copy()
xbg2 = XGBRegressor(n_estimators=500, early_stopping_rounds=5, learning_rate=0.1)
xbg2.fit(X_train_2, y_train, eval_set=[(X_valid[["num__sttl","cat__state_FIN","cat__state_REQ"]], y_valid)], verbose=False)

validation_xbg2 = xbg2.predict(X_valid[["num__sttl","cat__state_FIN","cat__state_REQ"]])
print("mean absolute error for XGBRegressor no.2: ", mean_absolute_error(validation_xbg2, y_valid))
