import numpy as np 
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb

train = pd.read_csv("train_bureau_raw.csv")
test = pd.read_csv("test_bureau_raw.csv")

app_train = pd.get_dummies(train)
app_test = pd.get_dummies(test)

train_labels = app_train['TARGET']
train_sk_id_curr = app_train['SK_ID_CURR']
test_sk_id_curr = app_test['SK_ID_CURR']

app_train.drop('SK_ID_CURR', inplace=True, axis=1)
app_test.drop('SK_ID_CURR', inplace=True, axis=1)

app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)
print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)

ratio = (train_labels == 0).sum()/ (train_labels == 1).sum()
ratio

X_train, X_test, y_train, y_test = train_test_split(app_train, train_labels, test_size=0.2, stratify=train_labels, random_state=1)
print("Postive examples in train set: {}".format(np.sum(y_train==0)))
print("Negative examples in train set: {}".format(np.sum(y_train==1)))

print("Postive examples in test set: {}".format(np.sum(y_test==0)))
print("Negative examples in test set: {}".format(np.sum(y_test==1)))


clf = XGBClassifier(n_estimators=1000, objective='binary:logistic', gamma=0.1, subsample=0.5,max_depth = 4,learning_rate = 0.05, scale_pos_weight=ratio )
clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='auc', early_stopping_rounds=30)


n_estimators = clf.best_ntree_limit
clf = XGBClassifier(n_estimators=n_estimators, objective='binary:logistic', gamma=0.1, subsample=0.5,max_depth = 4,learning_rate = 0.05, scale_pos_weight = ratio )
clf.fit(app_train.values, train_labels.values, eval_set=[(app_train.values, train_labels.values)], eval_metric='auc')

predictions = clf.predict_proba(app_test.values)[:, 1]
submission = pd.DataFrame({'SK_ID_CURR': test_sk_id_curr.values, 'TARGET': predictions})
submission.to_csv('xgboost_2.csv', index = False)