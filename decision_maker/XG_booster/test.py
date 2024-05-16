import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

# 테스트 데이터 불러오기
test_df = pd.read_csv("data/test.csv")

# 테스트 데이터에서 레이블 추출
X_test = test_df.drop(columns=["tech_approval_required"])
y_test = test_df["tech_approval_required"]

# XGBoost DMatrix 생성
dtest = xgb.DMatrix(X_test, label=y_test)

# 모델 불러오기
model = xgb.Booster()
model.load_model("xgboost_model.model")

# 테스트 데이터로 예측
test_preds = model.predict(dtest)

# AUC 스코어 계산
auc_score_test = roc_auc_score(y_test, test_preds)
print("Test AUC Score:", auc_score_test)