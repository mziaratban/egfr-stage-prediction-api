import streamlit as st
import requests
import numpy as np
import pandas as pd

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="eGFR Predictor", layout="wide")

st.title("eGFR Prediction Dashboard")
st.markdown("---")

# -------------------------
# Drug mapping
# -------------------------
raw_drugs = [
    'Cyclosporine', 'Valsartan', 'Ramipril', 'Pantoprazole', 'Olmesartan',
    'Tenofovir disoproxil', 'Chlorthalidone', 'Piperacillin', 'Indomethacin', 'Cidofovir',
    'Zoledronic acid', 'Spironolactone', 'Allopurinol', 'Trimethoprim', 'Ketorolac',
    'Amikacin', 'Celecoxib', 'Cimetidine', 'Vancomycin', 'Cisplatin',
    'Trandolapril', 'Metolazone', 'Foscarnet', 'Ciprofloxacin', 'Methotrexate',
    'Cephalexin', 'Enalapril', 'Diclofenac', 'Amiloride', 'Nafcillin',
    'Pemetrexed', 'Losartan', 'Amphotericin B', 'Warfarin', 'Tobramycin',
    'Furosemide', 'Eplerenone', 'Lisinopril', 'Clopidogrel', 'Meropenem',
    'Acyclovir', 'Naproxen', 'Gentamicin', 'Colistin', 'Meloxicam',
    'Tacrolimus', 'Quinapril', 'Bumetanide', 'Carboplatin', 'Telmisartan',
    'Hydrochlorothiazide', 'Irbesartan', 'Rifampin', 'Ibuprofen', 'Amoxicillin',
    'Heparin', 'Levofloxacin', 'Captopril', 'Ceftriaxone', 'Enoxaparin',
    'Iohexol', 'Cefepime', 'Dronedarone', 'Rivaroxaban', 'Apixaban',
    'Ticagrelor', 'Dolutegravir', 'Empagliflozin', 'Cobicistat'
]

indices = [
    23, 24, 25, 26, 30, 31, 33, 34, 35, 37, 39, 41, 42, 43, 45, 46, 47, 50, 51,
    52, 53, 54, 55, 56, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 72,
    75, 76, 77, 78, 80, 81, 83, 84, 86, 87, 90, 91, 94, 95, 96, 97, 98, 99, 101,
    102, 103, 104, 106, 107, 110, 111, 112, 113, 115, 116, 117
]

paired = sorted(zip(raw_drugs, indices), key=lambda x: x[0])
drug_mapping = dict(paired)
sorted_drugs = [d for d, _ in paired]

# -------------------------
# Sidebar inputs
# -------------------------
with st.sidebar:
    st.header("📋 Patient Demographics")

    age = st.number_input("Age (years)", 1, 120, 60, step=10)
    gender = st.radio("Gender", ["Male", "Female"])

    st.markdown("---")
    st.header("📋 Lab Items")

    scr = st.number_input("Serum Creatinine (mg/dL)", 0.0, 25.0, 3.0, step=0.05)
    urea = st.number_input("Urea (mg/dL)", 0.0, 400.0, 50.0, step=5.0)

    st.markdown("---")
    submit_button = st.button("Run Prediction Model", type="primary", use_container_width=True)

# -------------------------
# Drug selection grid
# -------------------------
st.subheader("💊 Administered medications in last five days")

cols = st.columns(8)
user_selections = {}

for i, drug in enumerate(sorted_drugs):
    with cols[i % 8]:
        user_selections[drug] = st.checkbox(drug, key=drug)

# -------------------------
# Prediction
# -------------------------
if submit_button:

    feature_vector = np.zeros(166, dtype=float)

    # engineered features
    feature_vector[5] = scr / 5
    feature_vector[11] = urea / 100
    feature_vector[164] = (age // 9) / 10
    feature_vector[165] = 0 if gender == "Male" else 1

    # drugs
    for drug, selected in user_selections.items():
        if selected:
            feature_vector[drug_mapping[drug]] = 1.0

    st.sidebar.success("Vector compiled successfully!")

    # -------------------------
    # IMPORTANT: Docker-safe URL
    # -------------------------
    API_URL = "http://fastapi:8000/predict"

    payload = {"features": feature_vector.tolist()}

    try:
        response = requests.post(API_URL, json=payload, timeout=30)

        if response.status_code != 200:
            st.error(f"Backend error: {response.status_code}")
        else:
            api_data = response.json()
            preds = np.array(api_data["predictions"])

            st.write("### Forecasting Results")
            st.write("Predicted eGFR stages over next 7 days:")

            df = pd.DataFrame(
                [preds],
                columns=[f"Day {i}" for i in range(1, 8)],
                index=["Stage"]
            )

            st.table(df)

    except requests.exceptions.RequestException as e:
        st.error(f"API connection failed: {str(e)}")