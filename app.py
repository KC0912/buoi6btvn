import streamlit as st
import numpy as np
import pickle
import os

# ======================
# LOAD MODEL
# ======================
model_path = os.path.join(os.path.dirname(__file__), "random_forest_model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

# ======================
# TITLE
# ======================
st.title("Academic Status Prediction")
st.write("Dự đoán Cảnh báo học vụ bằng Random Forest")

# ======================
# INPUT DATA
# ======================
age = st.number_input("Age", 15, 60, 20)
count_f = st.number_input("Count_F (Số môn F)", 0, 20, 0)
tuition_debt = st.number_input("Tuition_Debt (Nợ học phí)", 0, 100000000, 0)

# ======================
# PREDICT
# ======================
if st.button("Predict Academic Status"):

    data = np.array([[age, count_f, tuition_debt]])

    pred = model.predict(data)[0]

    status_map = {
        0: "Normal",
        1: "Academic Warning",
        2: "Dropout"
    }

    result = status_map.get(pred, pred)

    if result == "Normal":
        st.success(f"Status: {result}")

    elif result == "Academic Warning":
        st.warning(f"Status: {result}")

    else:
        st.error(f"Status: {result}")
