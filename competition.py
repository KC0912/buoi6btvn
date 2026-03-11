import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =======================
# 1. LOAD TRAIN DATA
# =======================
train_df = pd.read_csv("train.csv")

X = train_df[['Age', 'Count_F', 'Tuition_Debt']]
y = train_df['Academic_Status']

# Drop NaN
X = X.dropna()
y = y.loc[X.index]

# =======================
# 2. TRAIN / TEST SPLIT (để kiểm tra)
# =======================
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =======================
# 3. TRAIN RANDOM FOREST
# =======================
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)

# =======================
# 4. EVALUATION
# =======================
y_pred = rf_model.predict(X_val)

print("===== RANDOM FOREST VALIDATION =====")
print("Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(
    y_val, y_pred,
    target_names=["Normal", "Academic Warning", "Dropout"]
))

# =======================
# 5. SAVE MODEL
# =======================
with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

# =======================
# 6. LOAD TEST DATA (KAGGLE)
# =======================
test_df = pd.read_csv("test.csv")

# ⚠️ ĐỔI TÊN CỘT ID NẾU CẦN
submission = pd.DataFrame()
submission["Student_ID"] = test_df["Student_ID"]

X_test = test_df[['Age', 'Count_F', 'Tuition_Debt']]
X_test = X_test.fillna(X_test.median())

# =======================
# 7. PREDICT & SUBMISSION
# =======================
submission["Academic_Status"] = rf_model.predict(X_test)

submission.to_csv("submission.csv", index=False)

print(" Đã tạo file submission.csv")
print(submission.head())
