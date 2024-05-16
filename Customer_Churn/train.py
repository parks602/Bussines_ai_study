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

dataset = 'churn_data.csv'

df = pd.read_csv(dataset)
columns = df.columns.tolist()
encoded_data = df.drop(['id', 'customer_code', 'co_name'], axis=1)
encoded_data.head()

y = encoded_data['churned']
train_df, test_and_val_data, _, _ = train_test_split(encoded_data, y, test_size=0.3, stratify=y, random_state=0)

y = test_and_val_data['churned']
val_df, test_df, _, _ = train_test_split(test_and_val_data, y, test_size=0.333, stratify=y, random_state=0)

if not os.path.exists('data'):
    os.makedirs('data')
    
train_data = train_df.to_csv('data/train.csv', header=False, index=False)
val_data = val_df.to_csv('data/val.csv', header=False, index=False)
test_data = test_df.to_csv('data/test.csv', header=True, index=False)
 
X_train = train_df.drop(columns=["churned"])
y_train = train_df["churned"]
X_val = val_df.drop(columns=["churned"])
y_val = val_df["churned"]

# XGBoost 모델의 설정
params = {
    "max_depth": 5,
    "subsample": 0.7,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "num_round": 100,
    "early_stopping_rounds": 10,
    "scale_pos_weight": 17
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