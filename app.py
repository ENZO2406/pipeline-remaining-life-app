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
if 'df_input' not in st.session_state: st.session_state.df_input = pd.DataFrame()
if 'df_result' not in st.session_state: st.session_state.df_result = None
if 'excel_data' not in st.session_state: st.session_state.excel_data = None

# --- PHASE SELECTION ---
phase_choice = st.selectbox("🔍 Select the Analysis Phase / Sélectionner la phase d'analyse", ["Phase 1", "Phase 2", "Phase 3"])

@st.cache_resource
def load_model(phase):
    file_mapping = {"Phase 1": "rf_phase_1_assets.pkl", "Phase 2": "rf_phase_2_assets.pkl", "Phase 3": "rf_phase_3_assets.pkl"}
    target = file_mapping.get(phase)
    if target and os.path.exists(target):
        try: return joblib.load(target)
        except: return None
    return None

assets = load_model(phase_choice)

# --- ENCODING & CLEANING ---
def encode_categorical_data(df):
    mapping = {
        'Pipe Type': {'ERW': 1, 'SAWH': 2, 'SAWL': 3, 'SEAMLESS': 4},
        'Pipe Position': {'ABOVEGROUND': 1, 'BURIED': 2, 'RAWA': 3, 'RIVER CROSSING': 4, 'ROAD CROSSING': 5},
        'Insulation ': {'YES': 1, 'NO': 0}, 
        'Fluid Representative': {'CONDENSATE': 1, 'FOUL FLUID': 2, 'GAS': 3, 'HEAVY OIL': 4, 'LIGHT OIL': 5, 'STEAM': 6, 'WATER': 7},
        '3oF': {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5},
        'Soil Resistivity': {'MILDLY': 1, 'MILDLY/MODERATELY': 2, 'MODERATELY': 3, 'VERY': 4, 'POOR/UNKNOWN': 5},
        'Coating': {'FBE': 1, '3LPE': 2, 'COAL TAR': 3, 'NONE': 0}
    }
    df_encoded = df.copy()
    for col, m in mapping.items():
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].astype(str).str.upper().str.strip().map(m).fillna(1.0)
    return df_encoded.apply(pd.to_numeric, errors='coerce').fillna(1.0)

# --- UI ---
tab1, tab2 = st.tabs(["📁 Upload Excel", "✍️ Manual Entry / Saisie manuelle"])

with tab1:
    file = st.file_uploader("Upload Excel file", type=["xlsx", "csv"])
    if file:
        st.session_state.df_input = pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)

with tab2:
    st.write("### ✍️ Manual Data Entry / Saisie manuelle (24 Parameters)")
    cols = st.columns(3)
    # Paramètres avec instructions bilingues
    params = [
        "NPS (inch)", "Nominal Thickness (mm) / Épaisseur nominale", "Minimum Thickness / Épaisseur min", 
        "Water Cut / Taux d'eau", "OverallCR", "Soil pH / pH du sol", "Design Pressure (psi) / Pression", 
        "Leak Count / Nombre de fuites", "Location Class / Classe emplacement", "Insulation (YES=1, NO=0) / Isolation", 
        "Pipe Type (1-4) / Type de tuyau", "Pipe Position (1-5) / Position", "Fluid Rep (1-7) / Fluide", 
        "3oF (1-5)", "Soil Resistivity (1-5) / Résistivité", "Coating (0-3) / Revêtement", 
        "Param17", "Param18", "Param19", "Param20", "Param21", "Param22", "Param23", "Param24"
    ]
    data_manual = {}
    for i, p in enumerate(params):
        data_manual[p] = cols[i % 3].text_input(p, "1.0")
    
    if st.button("💾 Load Manual Data / Charger données"):
        st.session_state.df_input = pd.DataFrame([data_manual])

# --- PREDICTION ---
if not st.session_state.df_input.empty:
    st.write("### Preview / Aperçu:")
    st.dataframe(st.session_state.df_input.head(2))
    
    if st.button("🚀 Run Prediction / Lancer la simulation"):
        if assets is None:
            st.error("Model file missing / Fichier modèle introuvable.")
        else:
            try:
                df_clean = encode_categorical_data(st.session_state.df_input)
                preds = assets['model'].predict(assets['scaler'].transform(df_clean))
                st.session_state.df_result = st.session_state.df_input.copy()
                st
