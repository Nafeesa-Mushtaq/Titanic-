"""
Titanic Passenger Survival Prediction - Student Demo App
----------------------------------------------------------
A small Streamlit web app that lets students interact with the trained
Support Vector Classifier from the "1a - Titanic Survival - Recommended
Update" notebook, without needing to install Python or run Jupyter.

It loads the same svc_trained_model.pkl produced by that notebook, and
recreates the exact same LabelEncoders (fit on the same fixed category
lists) used during training, so predictions match the notebook exactly.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

# ---------------------------------------------------------------------------
# Load the trained model (same file the notebook produced)
# ---------------------------------------------------------------------------

MODEL_PATH = Path(__file__).parent / "svc_trained_model.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def build_label_encoders():
    """
    Recreates the exact same LabelEncoders used in the notebook.
    LabelEncoder assigns codes based on the sorted order of the category
    list, so fitting on the same fixed lists reproduces the same codes
    deterministically - no need to pickle/ship the encoders separately.
    """
    encoders = {}
    encoders["PClass"] = LabelEncoder().fit(["First", "Second", "Third"])
    encoders["Gender"] = LabelEncoder().fit(["Male", "Female"])
    encoders["Sibling"] = LabelEncoder().fit(["Zero", "One", "Two", "Three"])
    encoders["Embarked"] = LabelEncoder().fit(["Southampton", "Cherbourg", "Queenstown"])
    return encoders

model = load_model()
encoders = build_label_encoders()

# ---------------------------------------------------------------------------
# Page content
# ---------------------------------------------------------------------------

st.title("🚢 Titanic Passenger Survival Predictor")
st.markdown(
    """
    This is a teaching demo built from the **Titanic Passenger Survival Prediction**
    notebook. Pick a passenger's details below and the trained Support Vector
    Classifier (SVC) will predict whether they would have survived.

    *Model accuracy on held-out test data: **80%** (100-row toy dataset).*
    """
)

st.divider()

col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Passenger Class (PClass)", ["First", "Second", "Third"], index=2)
    gender = st.selectbox("Gender", ["Male", "Female"])
with col2:
    sibling = st.selectbox("Siblings/Spouses Aboard", ["Zero", "One", "Two", "Three"])
    embarked = st.selectbox("Port of Embarkation", ["Southampton", "Cherbourg", "Queenstown"])

if st.button("Predict Survival", type="primary", use_container_width=True):
    user_input = pd.DataFrame({
        "PClass": [pclass],
        "Gender": [gender],
        "Sibling": [sibling],
        "Embarked": [embarked],
    })

    encoded = pd.DataFrame({
        "PClass": encoders["PClass"].transform(user_input["PClass"]),
        "Gender": encoders["Gender"].transform(user_input["Gender"]),
        "Sibling": encoders["Sibling"].transform(user_input["Sibling"]),
        "Embarked": encoders["Embarked"].transform(user_input["Embarked"]),
    })

    prediction = model.predict(encoded)[0]

    st.divider()
    if prediction == 1:
        st.success("### ✅ SURVIVED")
    else:
        st.error("### ❌ NOT SURVIVED")

    with st.expander("See the encoded feature vector sent to the model"):
        st.dataframe(encoded, hide_index=True)

st.divider()
st.caption(
    "Built for teaching purposes from a 100-row simplified Titanic dataset. "
    "Predictions reflect patterns in this small toy dataset, not a rigorous "
    "historical analysis."
)
