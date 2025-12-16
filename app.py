#import all the neccessary libraries
# python -m streamlit run app.py

import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st


# filenames
MODEL_FILE = "pollution_model.pkl"
COLS_FILE = "model_columns.pkl"


def try_load_joblib(path):
    return joblib.load(path)


# UI header
st.title("Water Pollutants Predictor")
st.write("Predict water pollutant levels by Year and Station ID")


# Load model columns (may be present in repo)
model_cols = None
if os.path.exists(COLS_FILE):
    try:
        model_cols = try_load_joblib(COLS_FILE)
    except Exception as e:
        st.error(f"Could not load '{COLS_FILE}': {e}")


# Load or upload model
model = None
if os.path.exists(MODEL_FILE):
    try:
        model = try_load_joblib(MODEL_FILE)
    except Exception as e:
        st.error(f"Failed to load '{MODEL_FILE}': {e}")
else:
    st.warning(f"Model file '{MODEL_FILE}' not found in the repository.")
    uploaded_model = st.file_uploader("Upload trained model (pollution_model.pkl)", type=["pkl"])
    if uploaded_model is not None:
        try:
            # save uploaded file and load it
            with open(MODEL_FILE, "wb") as f:
                f.write(uploaded_model.getbuffer())
            model = try_load_joblib(MODEL_FILE)
            st.success("Model uploaded and loaded successfully.")
        except Exception as e:
            st.error(f"Uploaded file could not be loaded as a joblib model: {e}")


# If model or model_cols are missing, show guidance
if model is None:
    st.info("To use this app, add `pollution_model.pkl` (the trained model) to the repository root, or upload it here.")
    st.stop()

if model_cols is None:
    st.info("`model_columns.pkl` not found or could not be loaded. The app needs this file to format inputs correctly.")
    st.stop()


# User inputs
year_input = st.number_input("Enter Year", min_value=2000, max_value=2100, value=2022)
station_id = st.text_input("Enter Station ID", value="1")


def prepare_input(year, station, cols):
    df = pd.DataFrame({"year": [year], "id": [station]})
    enc = pd.get_dummies(df, columns=["id"])
    for c in cols:
        if c not in enc.columns:
            enc[c] = 0
    enc = enc[cols]
    return enc


if st.button("Predict"):
    if not station_id:
        st.warning("Please enter the station ID")
    else:
        input_encoded = prepare_input(year_input, station_id, model_cols)
        try:
            predicted_pollutants = model.predict(input_encoded)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
        else:
            pollutants = ["O2", "NO3", "SO4", "PO4", "CL"]
            st.subheader(f"Predicted pollutant level for station '{station_id}' in {year_input}:")
            for p, val in zip(pollutants, predicted_pollutants):
                st.write(f'{p}: {val:.2f}')


