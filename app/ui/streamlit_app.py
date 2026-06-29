import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="eGFR Predictor", layout="wide")

st.title("eGFR Prediction Dashboard")
#st.markdown("---")
#st.divider()
st.markdown( '<div style="border-top: 1px solid #e6e9ef; margin: 1px 0;"></div>',   unsafe_allow_html=True)

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
# with st.sidebar:
#     st.header("📋 Patient Demographics")

#     age = st.number_input("Age (years)", 1, 120, 60, step=10)
#     gender = st.radio("Gender", ["Male", "Female"])

#     st.markdown("---")
#     st.header("📋 Lab Items")

#     scr = st.number_input("Serum Creatinine (mg/dL)", 0.0, 25.0, 3.0, step=0.05)
#     urea = st.number_input("Urea (mg/dL)", 0.0, 400.0, 50.0, step=5.0)

#     st.markdown("---")
#     submit_button = st.button("Run Prediction Model", type="primary", use_container_width=True)


with st.sidebar:
    st.header("📋 Patient Demographics")

    age = st.number_input("Age (years)", 1, 120, 60, step=10)
    gender = st.radio("Gender", ["Male", "Female"])

    #st.markdown("---")
    #st.divider()
    st.markdown( '<div style="border-top: 1px solid #c6c9cf; margin: 1px 0;"></div>',   unsafe_allow_html=True)
    
    st.header("📋 Lab Items")

    # 1. Serum Creatinine Input & Conversion
    scr = st.number_input(
        "Serum Creatinine (mg/dL)", 0.0, 25.0, 3.0, step=0.05
    )

    # Conversion calculation (mg/dL to umol/L)
    scr_umol = scr * 88.4
    # st.caption(f"💡 Converted: **{scr_umol:.1f}** μmol/L")
    st.caption(f" ➜ {scr_umol:.1f} μmol/L")
    #st.write(f"   {scr_umol:.1f} μmol/L")

    st.write("")  # Adds a tiny bit of spacing

    # 2. Urea Input & Conversion
    urea = st.number_input("Urea (mg/dL)", 0.0, 400.0, 50.0, step=5.0)

    # Conversion calculation (mg/dL to mmol/L)
    # Note: Using the standard full Blood Urea conversion factor (0.166).
    # If your input is actually BUN (Blood Urea Nitrogen), change 0.166 to 0.357.
    urea_mmol = urea * 0.166
    st.caption(f" ➜ {urea_mmol:.1f} mmol/L")
    #st.info(f"   {urea_mmol:.1f} mmol/L")

    st.markdown("---")
    # st.divider()
    #st.markdown( '<div style="border-top: 1px solid #b6b9bf; margin: 10px 0;"></div>',   unsafe_allow_html=True)
    
    submit_button = st.button(
        "Run Prediction Model", type="primary", use_container_width=True
    )
    
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

    #st.sidebar.success("Vector compiled successfully!")

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
            
            st.markdown( '<div style="border-top: 1px solid #e6e9ef; margin: 1px 0;"></div>',   unsafe_allow_html=True)

            st.write("### Forecasting Results")
            st.write("Predicted eGFR stages over next 7 days:")


            df = pd.DataFrame(
                [preds], columns=[f"Day {i}" for i in range(1, 8)], index=["Stage"]
            )

            # --- 2. Create your custom 5-step color gradient ---
            custom_hex_colors = [
                "#2ca02c",  # 1: Green (Normal)
                "#a1d99b",  # 2: Light Green
                "#fcfd00",  # 3: Yellow
                "#ff7f0e",  # 4: Orange
                "#d62728",  # 5: Red (Abnormal)
            ]

            # Build a smooth linear colormap out of your custom sequence
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "severity_cmap", custom_hex_colors
            )

            
            stage_labels = {
                1: "Normal",
                2: "Mild",
                3: "Moderate",
                4: "Severe",
                5: "Failure",
            }            
            
            # --- 3. Apply the Style ---
            styled_df = (
                df.style.background_gradient(cmap=cmap, vmin=1, vmax=5)
                # Center align text inside the data cells
                .set_properties(
                    **{
                        "text-align": "center",
                        "padding": "5px",
                        "font-size": "16px !important"
                    }
                ).set_table_styles(
                    [
                        # Force the main table container to expand 100% horizontally
                        {
                            "selector": "",  # Empty selector targets the root <table> element
                            "props": [
                                ("width", "100% !important"),
                                ("table-layout", "fixed !important"),
                            ],
                        },
                        # Center align and format the header cells (th)
                        {
                            "selector": "th",
                            "props": [
                                ("text-align", "center !important"),
                                (
                                    "background-color",
                                    "#f2f4f8 !important",
                                ),  # Gray background
                                ("color", "#31333F !important"),
                                ("font-weight", "bold !important"),
                                ("padding", "5px !important"),
                                ("font-size", "14px !important")
                            ],
                        },
                    ]
                )
                # Substitute numeric values with "Number: Label" text dynamically
                .format(lambda val: f"{int(val)}: {stage_labels.get(int(val), '')}")
            )

            # --- 4. Render via Pure HTML ---
            html_table = styled_df.to_html()
            # Wrap the table in a div that clips the sharp corners and adds a clean border
            rounded_html_wrapper = f"""
            <div style="
                border: 1px solid #e0e2e6; 
                border-radius: 8px; 
                overflow: hidden; 
                width: 100%;
            ">
                {html_table}
            </div>
            """

            # Inject the rounded table structure directly into Streamlit
            st.html(rounded_html_wrapper)            
            
            
            
            
    except requests.exceptions.RequestException as e:
        st.error(f"API connection failed: {str(e)}")