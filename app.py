import streamlit as st
import requests
import numpy as np
import pandas as pd

# Set page to wide mode to maximize horizontal space
st.set_page_config(page_title="eGFR Predictor", layout="wide")
#🏥
st.title("eGFR Prediction Dashboard")
st.markdown("---")
# --- 1. DEFINE DRUG TO INDEX MAPPING ---
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



import joblib
model_1 = joblib.load("egfr_stage_model_day_1.pkl")
model_2 = joblib.load("egfr_stage_model_day_2.pkl")
model_3 = joblib.load("egfr_stage_model_day_3.pkl")
model_4 = joblib.load("egfr_stage_model_day_4.pkl")
model_5 = joblib.load("egfr_stage_model_day_5.pkl")
model_6 = joblib.load("egfr_stage_model_day_6.pkl")
model_7 = joblib.load("egfr_stage_model_day_7.pkl")




# Pair them up, sort alphabetically by drug name, and convert back to lists
paired_drugs = sorted(zip(raw_drugs, indices), key=lambda x: x[0])
sorted_drugs = [pair[0] for pair in paired_drugs]
drug_mapping = dict(paired_drugs)

# --- 2. DEMOGRAPHICS SIDEBAR / COMPACT PANEL ---
with st.sidebar:
    st.header("📋 Patient Demographics")
    age = st.number_input("Age", min_value=1, max_value=120, value=60)
    gender = st.radio("Gender", options=["Male", "Female"])
    st.markdown("---")
    st.header("📋 Lab Items")
    scr = st.number_input("Serum Creatinine (SCr)", min_value=0.0, max_value=25.0, value=3.0, step=0.05)
    urea = st.number_input("Urea", min_value=0.0, max_value=400.0, value=50.0, step=5.0)
    
    st.markdown("---")
    #🚀
    submit_button = st.button("Run Prediction Model", type="primary", use_container_width=True)

# --- 3. THE COMPACT 6-COLUMN GRID FOR SORTED CHECKBOXES ---
st.subheader("💊 Administered medications in last five days")

columns = st.columns(8)
user_selections = {}

for i, drug in enumerate(sorted_drugs):
    col_index = i % 8
    with columns[col_index]:
        user_selections[drug] = st.checkbox(drug, key=drug)

# --- 4. PROCESSING LOGIC ON SUBMIT ---
if submit_button:
    # Initialize your baseline 166 vector
    feature_vector = np.zeros(166)
    
    # Apply mathematical scaling guidelines
    feature_vector[5] = scr / 5
    feature_vector[11] = urea / 100
    feature_vector[164] = (age // 9) / 10
    feature_vector[165] = 0 if gender == "Male" else 1
    
    # Map selected checkboxes directly to their specified array indices
    for drug, is_checked in user_selections.items():
        if is_checked:
            target_index = drug_mapping[drug]
            feature_vector[target_index] = 1.0

    st.sidebar.success("Vector compiled successfully!")
    
    # Send to your API endpoint
    API_URL = "http://127.0.0.1:8000/predict"
    payload = {"features": feature_vector.tolist()}
    
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            api_data = response.json()
            
            # --- CUSTOM RESULT DISPLAY WITH TABLE ---
            #🎯
            #st.markdown("---")
            st.write("###  Forecasting Results")
            st.write("**The forecasted eGFR stages in the following days are:**")
            features_2d = feature_vector.reshape(1, -1)
            
            # Assuming your API returns a list/array of 6 values under a key (e.g., 'predictions')
            # Example fallback dummy array if API layout differs: [3, 3, 4, 4, 4, 5]
            #predicted_stages = api_data.get("predictions", [-3, -3, -3, -4, -4, -4]) 
            
            raw_prediction = model_1.predict(features_2d)
            p_1 = int(raw_prediction[0])
            raw_prediction = model_2.predict(features_2d)
            p_2 = int(raw_prediction[0])
            raw_prediction = model_3.predict(features_2d)
            p_3 = int(raw_prediction[0])
            raw_prediction = model_4.predict(features_2d)
            p_4 = int(raw_prediction[0])
            raw_prediction = model_5.predict(features_2d)
            p_5 = int(raw_prediction[0])
            raw_prediction = model_6.predict(features_2d)
            p_6 = int(raw_prediction[0])
            raw_prediction = model_7.predict(features_2d)
            p_7 = int(raw_prediction[0])
            
            predicted_stages = np.array([p_1, p_2, p_3, p_4, p_5, p_6, p_7])
            #predicted_stages = np.array([feature_vector[5], feature_vector[11], feature_vector[-2], feature_vector[-1], feature_vector[23], feature_vector[24]])
            # Construct a clean DataFrame representing the two rows
            result_df = pd.DataFrame(
                [predicted_stages], 
                columns=["1 Days", "2 Days", "3 Days", "4 Days", "5 Days", "6 Days", "7 Days"],
                index=["Predicted eGFR Stage"]
            )
            
            # Display it as a wide table on the screen
            st.table(result_df)
            
        else:
            st.error(f"Backend returned error code: {response.status_code}")
    except Exception as e:
        st.error(f"Failed to reach API server: {e}")