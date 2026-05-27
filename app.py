import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
from io import BytesIO

warnings.filterwarnings('ignore', category=UserWarning)

st.set_page_config(page_title="Pipeline Reliability App", layout="wide")
st.title("📊 Pipeline Reliability Prediction App")
st.markdown("---")

# --- APP MEMORY ---
if 'df_input' not in st.session_state:
    st.session_state.df_input = pd.DataFrame()
if 'df_result' not in st.session_state:
    st.session_state.df_result = None
if 'excel_data' not in st.session_state:
    st.session_state.excel_data = None

# --- PHASE SELECTION MENU ---
phase_choice = st.selectbox("🔍 Select the Analysis Phase", ["Phase 1", "Phase 2", "Phase 3"])

# 1. DYNAMIC LOAD MODEL BASED ON SELECTION
@st.cache_resource
def load_model(phase):
    file_mapping = {
        "Phase 1": "rf_phase_1_assets.pkl",
        "Phase 2": "rf_phase_2_assets.pkl",
        "Phase 3": "rf_phase_3_assets.pkl"
    }
    target_file = file_mapping[phase]
    
    if os.path.exists(target_file):
        try:
            return joblib.load(target_file)
        except Exception as e:
            st.error(f"Error loading the model ({target_file}): {e}")
            return None
    return None

assets = load_model(phase_choice)

# 2. ENCODING FUNCTION
def encode_categorical_data(df):
    encoding_maps = {
        'Pipe Type': {'ERW': 1, 'SAWH': 2, 'SAWL': 3, 'SEAMLESS': 4},
        'Pipe Position': {'ABOVEGROUND': 1, 'BURIED': 2, 'RAWA': 3, 'RIVER CROSSING': 4, 'ROAD CROSSING': 5},
        'Insulation ': {'YES': 1, 'NO': 0}, 
        'Fluid Representative': {'CONDENSATE': 1, 'FOUL FLUID': 2, 'GAS': 3, 'HEAVY OIL': 4, 'LIGHT OIL': 5, 'STEAM': 6, 'WATER': 7},
        '3oF': {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5},
        'Soil Resistivity': {'MILDLY': 1, 'MILDLY/MODERATELY': 2, 'MODERATELY': 3, 'VERY': 4, 'POOR/UNKNOWN': 5},
        'Coating': {'FBE': 1, '3LPE': 2, 'COAL TAR': 3, 'NONE': 0}
    }
    
    df_encoded = df.copy()
    for col, mapping in encoding_maps.items():
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].astype(str).str.upper().str.strip()
            df_encoded[col] = df_encoded[col].map(mapping).fillna(1.0)
    return df_encoded

# 3. USER INTERFACE
file = st.file_uploader("📁 Upload the Excel data file", type=["xlsx", "csv"])

if file:
    st.session_state.df_input = pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)
    st.session_state.df_result = None
    st.session_state.excel_data = None
    
if not st.session_state.df_input.empty:
    df_input = st.session_state.df_input
    st.write("### 📋 Preview of imported data:")
    st.dataframe(df_input.head(3))
    
    if st.button("🚀 Run Diagnostic and Prediction", type="primary"):
        if assets is None:
            st.error(f"The required model file for {phase_choice} is missing from the server.")
        else:
            try:
                with st.spinner(f"Processing data using {phase_choice}..."):
                    # --- STEP A: EXACT MAPPING ---
                    col_mapping = {
                        'NPS (inch)': 'NPS (inch)',
                        'Nominal Thickness (mm)': 'Nominal_Thickness',
                        'Minimum Thickness': 'Minimum_Thickness',
                        'Insulation (YES/NO)': 'Insulation ',
                        'Water Cut': 'Water_Cut',
                        'OverallCR': 'OverallCR', 
                        'Soil pH': 'Soil_pH',
                        'CoF': '3oF',
                        'Design Pressure (psi)': 'Design_Pressure',
                        'Leak Count': 'Leak_Count'
                    }
                    df_prepared = df_input.rename(columns=col_mapping)
                    df_encoded = encode_categorical_data(df_prepared)
                    
                    # --- STEP B: FORMULA CALCULATIONS (SPLIT BY PHASE) ---
                    if phase_choice == "Phase 1" or phase_choice == "Phase 2":
                        # Phase 1 & Phase 2 both require engineering calculations
                        t_nom = pd.to_numeric(df_encoded.get('Nominal_Thickness', 12.7), errors='coerce').fillna(12.7)
                        t_min = pd.to_numeric(df_encoded.get('Minimum_Thickness', 9.5), errors='coerce').fillna(9.5)
                        D = pd.to_numeric(df_encoded.get('NPS (inch)', 508.0), errors='coerce').fillna(508.0)
                        
                        L = t_nom * 0.20
                        d = t_nom - t_min 
                        
                        df_encoded['severity_ratio'] = np.where(t_nom > 0, d / t_nom, 0)
                        
                        z = np.where((D * t_nom) > 0, (L**2) / (D * t_nom), 1.0)
                        M = np.where(z <= 50, 0.032 * z + 3.3, np.sqrt(1 + 0.48 * z - 0.003375 * (z**2)))
                        num = 1 - 0.85 * np.where(t_nom > 0, d / t_nom, 0)
                        den = 1 - 0.85 * np.where((M * t_nom) > 0, d / (M * t_nom), 1)
                        df_encoded['asme_b31g'] = np.where(den != 0, num / den, 1.0)

                    elif phase_choice == "Phase 3":
                        # Phase 3 does NOT use Soil_pH nor calculated parameters. No equations needed.
                        pass
                    
                    # --- STEP C: FINAL FILTERING FOR THE MODEL ---
                    model = assets['model']
                    scaler = assets['scaler']
                    expected_features = assets['feature_names']
                    
                    final_df = pd.DataFrame(index=df_encoded.index)
                    
                    for col in expected_features:
                        if col in df_encoded.columns:
                            final_df[col] = pd.to_numeric(df_encoded[col], errors='coerce')
                        else:
                            final_df[col] = 1.0 
                            
                    final_df = final_df.fillna(1.0)
                    
                    # --- DIAGNOSTIC DISPLAY ---
                    st.warning(f"⚠️ **DIAGNOSTIC TABLE ({phase_choice}):** Exact numbers sent to the model. Check for abnormal 1.0 values.")
                    st.dataframe(final_df)
                    
                    # --- STEP D: PREDICTION ---
                    scaled_data = scaler.transform(final_df)
                    predictions = model.predict(scaled_data)
                    
                    # --- SAVE TO MEMORY FOR THE BUTTON ---
                    df_result = df_input.copy()
                    df_result.insert(0, 'ESTIMATED LIFE (YEARS)', np.round(predictions, 2))
                    df_result.insert(1, 'APPLIED MODEL', phase_choice)
                    
                    st.session_state.df_result = df_result
                    
                    # Create Excel buffer in memory
                    excel_buffer = BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        df_result.to_excel(writer, index=False, sheet_name='Predictions')
                    st.session_state.excel_data = excel_buffer.getvalue()
                    
            except Exception as e:
                st.error(f"An error occurred during the calculation: {e}")

# --- DISPLAY RESULTS AND EXCEL BUTTON ---
if st.session_state.df_result is not None:
    st.success("✅ Calculations completed successfully!")
    st.write("### 📈 Final Results:")
    st.dataframe(st.session_state.df_result)

    st.download_button(
        label="📥 Download Results (Excel File)",
        data=st.session_state.excel_data,
        file_name=f"predictions_{phase_choice.lower().replace(' ', '_')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary"
    )