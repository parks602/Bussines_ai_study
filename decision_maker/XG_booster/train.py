import pandas as pd
from time import sleep, time
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.datasets import dump_svmlight_file
import os
import warnings

warnings.filterwarnings('ignore')

dataset = 'orders_with_predicted_value.csv'

df = pd.read_csv(dataset)

encoded_data = pd.get_dummies(df)

print(encoded_data.head())
corrs = encoded_data.corr()['tech_approval_required'].abs()
columns = corrs[corrs > .1].index
corrs = corrs.filter(columns)

encoded_data = encoded_data[columns]


train_df, val_and_test_data = train_test_split(encoded_data, test_size=0.3, random_state=0)
val_df, test_df = train_test_split(val_and_test_data, test_size=0.333, random_state=0)

if not os.path.exists:
    os.makedirs('data')
train_data = train_df.to_csv('data/train.csv', header=False, index=False)
val_data = val_df.to_csv('data/val.csv', header=False, index=False)
test_data = test_df.to_csv('data/test.csv', header=True, index=False)
 
X_train = train_df.drop(columns=["tech_approval_required"])
y_train = train_df["tech_approval_required"]
X_val = val_df.drop(columns=["tech_approval_required"])
y_val = val_df["tech_approval_required"]

# XGBoost 모델의 설정
params = {
    "max_depth": 5,
    "subsample": 0.7,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "num_round": 100,
    "early_stopping_rounds": 10
}

# XGBoost DMatrix 생성
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# XGBoost 모델 학습
model = xgb.train(params, dtrain, evals=[(dtrain, "train"), (dval, "validation")])

# 검증 데이터로 AUC 스코어 계산
val_preds = model.predict(dval)
auc_score = roc_auc_score(y_val, val_preds)
print("Validation AUC Score:", auc_score)

# 학습된 모델 저장
model.save_model("xgboost_model.model")