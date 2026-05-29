import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
from io import BytesIO
import xlsxwriter

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
phase_choice = st.selectbox(
    "🔍 Select the Analysis Phase", 
    [
        "Phase 1 (Without Service_age)", 
        "Phase 2 (Without Service_age and Soil_pH)", 
        "Phase 3 (Without Service_age, Soil_pH, asme_b31g, combinatorial_effect, severity_ratio)"
    ]
)

# Sécurisation de la clé pour charger le modèle sans bug de texte
phase_key = phase_choice.split(" (")[0].strip()

# 1. DYNAMIC LOAD MODEL BASED ON SELECTION
@st.cache_resource
def load_model(phase):
    file_mapping = {
        "Phase 1": "rf_phase_1_assets.pkl",
        "Phase 2": "rf_phase_2_assets.pkl",
        "Phase 3": "rf_phase_3_assets.pkl"
    }
    target_file = file_mapping.get(phase)
    
    if target_file and os.path.exists(target_file):
        try:
            return joblib.load(target_file)
        except Exception as e:
            st.error(f"Error loading the model ({target_file}): {e}")
            return None
    return None

assets = load_model(phase_key)

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
tab1, tab2 = st.tabs(["📁 Upload Excel", "✍️ Manual Entry"])

with tab1:
    file = st.file_uploader("📁 Upload the Excel data file", type=["xlsx", "csv"])
    if file:
        st.session_state.df_input = pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)
        st.session_state.df_result = None
        st.session_state.excel_data = None

with tab2:
    st.write("### ✍️ Manual Data Entry (Enter values directly)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        nps = st.text_input("NPS (inch)", "508.0")
        nom_t = st.text_input("Nominal Thickness (mm)", "12.7")
        min_t = st.text_input("Minimum Thickness", "9.5")
        wc = st.text_input("Water Cut", "0.0")
        ocr = st.text_input("OverallCR", "0.1")
        ph = st.text_input("Soil pH", "7.0")
        pres = st.text_input("Design Pressure (psi)", "1000.0")
        leak = st.text_input("Leak Count", "0.0")
        
    with col2:
        loc_class = st.text_input("Location Class", "1.0")
        ins = st.text_input("Insulation (YES=1, NO=0)", "YES")
        pt = st.text_input("Pipe Type (ERW, SAWH, SAWL, SEAMLESS)", "ERW")
        pp = st.text_input("Pipe Position (ABOVEGROUND, BURIED, RAWA, RIVER CROSSING, ROAD CROSSING)", "ABOVEGROUND")
        fluid = st.text_input("Fluid Rep (CONDENSATE, GAS, WATER, etc.)", "GAS")
        cof = st.text_input("3oF (A, B, C, D, E)", "A")
        soil = st.text_input("Soil Resistivity (MILDLY, MODERATELY, VERY, POOR/UNKNOWN)", "MILDLY")
        coat = st.text_input("Coating (FBE, 3LPE, COAL TAR, NONE)", "FBE")

    with col3:
        var17 = st.text_input("Parameter 17", "1.0")
        var18 = st.text_input("Parameter 18", "1.0")
        var19 = st.text_input("Parameter 19", "1.0")
        var20 = st.text_input("Parameter 20", "1.0")
        var21 = st.text_input("Parameter 21", "1.0")
        var22 = st.text_input("Parameter 22", "1.0")
        var23 = st.text_input("Parameter 23", "1.0")
        var24 = st.text_input("Parameter 24", "1.0")

    if st.button("💾 Load Manual Data"):
        data = {
            'NPS (inch)': nps, 'Nominal Thickness (mm)': nom_t, 'Minimum Thickness': min_t,
            'Insulation (YES/NO)': ins, 'Water Cut': wc, 'OverallCR': ocr, 'Soil pH': ph,
            'Design Pressure (psi)': pres, 'Leak Count': leak, 'Location Class': loc_class,
            'Pipe Type': pt, 'Pipe Position': pp, 'Fluid Representative': fluid,
            '3oF': cof, 'Soil Resistivity': soil, 'Coating': coat,
            'Param17': var17, 'Param18': var18, 'Param19': var19, 'Param20': var20,
            'Param21': var21, 'Param22': var22, 'Param23': var23, 'Param24': var24
        }
        st.session_state.df_input = pd.DataFrame([data])
        st.success("Manual data loaded!")

if not st.session_state.df_input.empty:
    df_input = st.session_state.df_input
    st.write("### 📋 Preview of imported data:")
    st.dataframe(df_input.head(3))
    
    if st.button("🚀 Run Diagnostic and Prediction", type="primary"):
        if assets is None:
            st.error(f"The required model file for {phase_key} is missing from the server.")
        else:
            try:
                with st.spinner(f"Processing data using {phase_key}..."):
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
                        'Leak Count': 'Leak_Count',
                        'Location Class': 'Location_Class'
                    }
                    df_prepared = df_input.rename(columns=col_mapping)
                    df_encoded = encode_categorical_data(df_prepared)
                    
                    if phase_key in ["Phase 1", "Phase 2"]:
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
                    
                    # Calcul des prédictions
                    predictions = model.predict(scaler.transform(final_df))
                    rounded_preds = np.round(predictions, 2)
                    
                    # 1. Dataset Original + Résultats
                    df_result = df_input.copy()
                    df_result.insert(0, 'ESTIMATED LIFE (YEARS)', rounded_preds)
                    st.session_state.df_result = df_result
                    
                    # 2. Dataset Transformé + Résultats
                    df_encoded_result = final_df.copy()
                    df_encoded_result.insert(0, 'ESTIMATED LIFE (YEARS)', rounded_preds)
                    
                    # Génération de l'Excel avec 2 onglets différents
                    excel_buffer = BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                        df_result.to_excel(writer, index=False, sheet_name='Original_Predictions')
                        df_encoded_result.to_excel(writer, index=False, sheet_name='Encoded_Predictions')
                    st.session_state.excel_data = excel_buffer.getvalue()
                    
            except Exception as e:
                st.error(f"An error occurred during the calculation: {e}")

if st.session_state.df_result is not None:
    st.success("✅ Calculations completed successfully!")
    st.write("### 📈 Final Results:")
    st.dataframe(st.session_state.df_result)

    st.download_button(
        label="📥 Download Complete Results (Multi-Sheet Excel File)",
        data=st.session_state.excel_data,
        file_name=f"predictions_{phase_key.lower().replace(' ', '_')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary"
    )
