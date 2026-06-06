import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
from io import BytesIO
import xlsxwriter
import string

warnings.filterwarnings('ignore', category=UserWarning)

st.set_page_config(page_title="Pipeline Remaining life prediction App", layout="wide")
st.title("📊 Pipeline Remaining life prediction App")
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

phase_key = phase_choice.split(" (")[0].strip()

# 1. DYNAMIC LOAD MODEL
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

# ============================================================================
# 3. USER INTERFACE (TABS)
# ============================================================================
tab_gen, tab_mod, tab1, tab2 = st.tabs([
    "ℹ️ General Information", 
    "🤖 Model Information", 
    "📁 Upload Excel", 
    "✍️ Manual Entry"
])

# ----------------------------------------------------------------------------
# TAB A : GENERAL INFORMATION
# ----------------------------------------------------------------------------
with tab_gen:
    st.markdown("## 📁 GENERAL INFORMATION")
    st.write("---")
    
    st.markdown("### 👤 Author & Supervisors")
    col_auth1, col_auth2 = st.columns(2)
    with col_auth1:
        st.info("""
        **🎓 Student:**
        * **Enzo Roinson**
        * *University / Program:* IUT of Saint-Malo / Bachelor in Maintenance and Industrial Engineering
        """)
    with col_auth2:
        st.info("""
        **💼 Host Company & Supervision:**
        * **Company:** Dago Engineering (Bali, Indonesia)
        * *Industrial Supervisors:* Mr. Zasya, Mr. Daffa
        * *Academic Supervisors:* Ms. Valérie Coste, Mr. Philippe Dauphin
        """)
        
    st.markdown("### 📝 Project Background")
    st.write("""
    Industrial pipelines are critical assets that require regular monitoring to prevent catastrophic failures. 
    However, physical inspections (e.g., ultrasonic testing, intelligent pigging) can be **extremely complex, expensive, and time-consuming**, 
    especially for buried or hard-to-reach sections.
    
    **Objective:** This project leverages historical inspection data from similar pipeline segments to build a Machine Learning model. 
    The goal is to predict the **Remaining Life** of uninspected pipelines without conducting physical measurements on site.
    """)
    
    st.markdown("### 💡 Solution Workflow & Business Impact")
    st.success("🔄 **Workflow:** Historical Inspection Data ➔ Data Preprocessing ➔ Random Forest Model ➔ Predictive Dashboard ➔ Remaining Life Prediction ➔ Optimized Maintenance Decisions")
    
    st.markdown("""
    **📊 Business Benefits:**
    * **Cost Reduction:** Avoids expensive physical machinery mobilization and excavation unless strictly required.
    * **Time Optimization:** Provides instant remaining life estimation instead of weeks of field operations.
    * **Safety Enhancement:** Early detection of high-risk corrosion zones.
    """)
    
    # --- EXACT FORMULAS AND CALCULATION DETAILS ---
    st.markdown("### 📜 Framework Standards & Mathematical Background")
    st.write("The application does not only rely on AI but also automatically computes standard engineering features during preprocessing, based on international codes:")
    
    st.markdown("#### **1. API 570 - Remaining Life Assessment**")
    st.write("First, we have to choose a safety limit, so I've selected the 20% of the nominal thickness. I chose this rate according to the API 570 standard. Then, we can use the following formula:")
    
    st.latex(r"Reserve = Nominal\ thickness - Safety\ limite")
    st.latex(r"Years = \frac{reserve}{OverallCR}")
    st.latex(r"Remaining\ life = Years - service\ age")
    
    with st.expander("📝 Show API 570 Calculation Example"):
        st.write("""
        * **Nominal thickness** = 9.5 mm
        * **Nominal thickness** = 9.5 mm
        * **Safety limit** = 20% of 9.5 mm = 1.9 mm
        * **Reserve** = 9.5 - 1.9 = **7.6 mm**
        
        * **Overall Corrosion Rate (OverallCR)** = 0.1 mm/year
        * **Estimated total years (Years)** = 7.6 / 0.1 = **76 years**
        
        * **Service age** = 20 years
        * **Remaining life** = 76 - 20 = **56 years**
        """)

    st.markdown("#### **2. Defect Severity Ratio**")
    st.write("The severity ratio is calculated to determine the criticality of the defect. This parameter shows the rate of deterioration. To calculate this parameter, I refer to the ASME B31G standard to find the appropriate calculation method. Therefore, I use the following method:")
    
    st.latex(r"Depth = Nominal\ thickness - Minimum\ Thickness")
    st.latex(r"Ratio = \frac{Depth}{Nominal\ Thickness}")

    st.markdown("#### **3. ASME B31G (Folias Factor & Remaining Strength Factor)**")
    
    st.write("Calculation of the Folias factor (M):")
    st.latex(r"z = \left(\frac{L^2}{D \cdot t}\right)")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("**For the short default ( $z < 50$ )**")
        st.latex(r"M = 0{,}032\left(\frac{L^2}{D \cdot t}\right) + 3{,}3")
    with col_m2:
        st.markdown("**For the long default ( $z > 50$ )**")
        st.latex(r"M = \sqrt{1 + 0{,}48\left(\frac{L^2}{D \cdot t}\right) - 0{,}003375\left(\frac{L^2}{D \cdot t}\right)^2}")

    st.markdown("""
    * **L** is the length of the defect
    * **D** is the pipe diameter
    * **t** is the nominal wall thickness of the pipe
    """)
    
    st.write("After calculating the **Folias factor (M)**, we can calculate the **RSF (Remaining Strength Factor)**. The RSF represents the ratio between the burst pressure of the corroded pipe and that of an intact pipe. We use the following formula:")
    st.latex(r"RSF = \frac{1 - 0{,}85\left(\frac{d}{t}\right)}{1 - 0{,}85\left(\frac{d}{M \cdot t}\right)}")
    st.markdown("*(Note: $d$ represents the Depth of the defect calculated previously)*")
    
    st.markdown("---")
    st.markdown("### 🖥️ How to use this App & Phase Selection")
    st.write("""
    This application allows you to evaluate pipelines either individually (**Manual Entry** tab) or in bulk by uploading a dataset (**Upload Excel** tab). 
    Before running the prediction, you **must select an Analysis Phase** at the top of the screen. The choice depends entirely on the data you currently have available:
    """)
    
    st.info("""
    🟢 **PHASE 1 (Highest Accuracy):** * **When to use:** You have access to a comprehensive dataset, including geometric properties, fluid data, and environmental data (like `Soil pH`). The exact `Service_Age` of the pipe is the only missing variable.
    * **Why:** This phase utilizes the maximum number of parameters and computed ASME ratios to provide the most precise Remaining Life estimation.
    
    🔵 **PHASE 2 (Moderate Accuracy):** * **When to use:** You are missing both the `Service_Age` AND the `Soil pH`. 
    * **Why:** Very common for aboveground pipelines where soil analysis is irrelevant, or when environmental historical records are lost.
    
    🟠 **PHASE 3 (Robust Fallback):** * **When to use:** You only have basic operational data. You are missing `Service_Age`, `Soil pH`, and specific geometric parameters required to compute complex features like `ASME B31G` and `Severity Ratio`.
    * **Why:** This is a simplified model designed to give a reasonable estimate even when critical inspection parameters are unavailable.
    """)

# ----------------------------------------------------------------------------
# TAB B : MODEL INFORMATION
# ----------------------------------------------------------------------------
with tab_mod:
    st.markdown("## 🤖 MODEL INFORMATION")
    st.write("---")
    
    st.markdown("### 📊 Parameters Overview")
    param_table = """
    | Parameter Group | Key Features | Description |
    | :--- | :--- | :--- |
    | **Geometry & Design** | NPS (inch), Nominal/Min Thickness, Design Pressure | Geometric constraints and operational limits. |
    | **Environmental** | Soil pH, Soil Resistivity, Coating | External corrosivity indicators. |
    | **Fluid Properties** | Water Cut, Fluid Representative, H2S, Flowrate | Internal corrosion risk factors. |
    | **Operational History** | Service Age, Leak Count, OverallCR, Internal CR, PoF | Track record of asset integrity decay. |
    """
    st.markdown(param_table)
    
    col_mod1, col_mod2 = st.columns(2)
    with col_mod1:
        st.markdown("### 🧠 Model Selection Rationale")
        st.write("""
        **Algorithm: Random Forest Regressor**
        * **High Performance:** Demonstrated a superior r² score and low MAE.
        * **No Overfitting:** Generalizes very well on unseen test data compared to baseline models.
        * **Complex Interactions:** Efficiently handles non-linear relationships.
        """)
    with col_mod2:
        st.markdown("### 📂 Dataset Overview")
        st.write("""
        * **Inputs:** 24 variables (Categorical & Numerical)
        * **Validation Method:** 5-Fold Cross-Validation (CV)
        * **Rows:** ~60600
        * **Sources:** In-line Inspection (ILI) data, Soil analysis
        """)

    st.markdown("### ⚙️ Technical Details & Hyperparameters (GridSearch)")
    st.markdown("""
    * 🟢 **Phase 1:** `n_estimators=200`, `max_depth=10`, `max_features='sqrt'`, `min_samples_leaf=5`, `min_samples_split=10`
    * 🔵 **Phase 2:** `n_estimators=150`, `max_depth=10`, `max_features='sqrt'`, `min_samples_leaf=7`, `min_samples_split=20`
    * 🟠 **Phase 3:** `n_estimators=100`, `max_depth=8`, `max_features='sqrt'`, `min_samples_leaf=10`, `min_samples_split=30`
    """)

    st.markdown("### 🏆 Model Performance Evaluation by Phase")
    st.markdown("""
    | Evaluation Metric | Phase 1 *(Without Service_age)* | Phase 2 *(Without Service_age, Soil_pH)* | Phase 3 *(Without Age, pH, asme_b31g, combinatorial, severity)* |
    | :--- | :---: | :---: | :---: |
    | **r² CV** | 0.9670 | 0.9615 | 0.9312 |
    | **r² Train** | 0.9693 | 0.9630 | 0.9334 |
    | **r² Val** | 0.9663 | 0.9601 | 0.9306 |
    | **r² Test** | 0.9675 | 0.9618 | 0.9350 |
    | **MAE (Years)** | 2.0336 | 2.2648 | 3.3504 |
    | **RMSE** | 3.7800 | 4.0983 | 5.3463 |
    | **MAPE** | 2.39% | 2.67% | 4.08% |
    """)
    
    st.info("""
    **📈 Evaluation Criteria (Lewis MAPE Interpretation Standards) :**
    * **< 10%** : Highly Accurate Forecasting ✅ *(Our model falls here)*
    * **10% - 20%** : Good Forecasting
    * **20% - 50%** : Reasonable Forecasting
    * **> 50%** : Inaccurate Forecasting
    """)
    
    st.markdown("### 🛠️ Technical ML Pipeline")
    st.markdown("""
    1. **Data Preprocessing:** Handling missing values, mapping Categorical to Numerical features, engineering new features (ASME severity ratios).
    2. **Model Training:** Random Forest with tuned hyperparameters and cross-validation.
    3. **Postprocessing:** Outputting predictions in **Years** for remaining life.
    """)

# ----------------------------------------------------------------------------
# TAB 1 : UPLOAD EXCEL (Template Generation with Dropdown Menus)
# ----------------------------------------------------------------------------
with tab1:
    st.markdown("### 📥 1. Download Blank Template")
    st.write("If you don't have a dataset ready, download this template. **Selection lists are integrated directly into the Excel file cells (drop-down menus)!**")
    
    output = BytesIO()
    workbook = xlsxwriter.Workbook(output, {'in_memory': True})
    worksheet = workbook.add_worksheet('Data_Template')
    
    template_columns = {
        'NPS (inch)': None,
        'Nominal Thickness (mm)': None,
        'Minimum Thickness': None,
        'Insulation (YES/NO)': ['YES', 'NO'],
        'Water Cut': None,
        'OverallCR': None,
        'Soil pH': None,
        'CoF': ['A', 'B', 'C', 'D', 'E'],
        'Design Pressure (psi)': None,
        'Leak Count': None,
        'Location Class': None,
        'Pipe Type': ['ERW', 'SAWH', 'SAWL', 'SEAMLESS'],
        'Pipe Position': ['BURIED', 'ABOVEGROUND', 'RAWA', 'RIVER CROSSING', 'ROAD CROSSING'],
        'Fluid Representative': ['GAS', 'WATER', 'CONDENSATE', 'FOUL FLUID', 'HEAVY OIL', 'LIGHT OIL', 'STEAM'],
        'Soil Resistivity': ['MILDLY', 'MILDLY/MODERATELY', 'MODERATELY', 'VERY'],
        'Coating': ['Poor/Unknown', 'Bad', 'Fair', 'Good'],
        'Flowrate (BFPD)': None,
        'Service_Age': None,
        'HCA': None,
        'Internal CR': None,
        'PoF': ['1', '2', '3', '4', '5'],
        'H2S': None
    }
    
    header_format = workbook.add_format({'bold': True, 'bg_color': '#1f4e78', 'font_color': 'white', 'align': 'center', 'border': 1})
    excel_cols = list(string.ascii_uppercase)
    
    for col_num, (col_name, dropdown_options) in enumerate(template_columns.items()):
        worksheet.write(0, col_num, col_name, header_format)
        worksheet.set_column(col_num, col_num, 22)
        
        if dropdown_options:
            col_letter = excel_cols[col_num]
            worksheet.data_validation(f'{col_letter}2:{col_letter}1000', {
                'validate': 'list',
                'source': dropdown_options
            })
            
    workbook.close()
    
    st.download_button(
        label="📄 Download Excel Template",
        data=output.getvalue(),
        file_name="Pipeline_Blank_Template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    st.markdown("---")
    st.markdown("### 📁 2. Upload Data")
    file = st.file_uploader("📁 Upload the Excel data file", type=["xlsx", "csv"])
    if file:
        st.session_state.df_input = pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)
        st.session_state.df_result = None
        st.session_state.excel_data = None

# ----------------------------------------------------------------------------
# TAB 2 : MANUAL ENTRY
# ----------------------------------------------------------------------------
with tab2:
    st.write("### ✍️ Manual Data Entry (24 Parameters)")
    
    # --- NEW: Dropdown menu explaining each parameter ---
    with st.expander("📖 View Parameter Definitions"):
        st.markdown("""
        | Parameter | Definition / Description |
        | :--- | :--- |
        | **NPS (inch)** | Nominal Pipe Size in inches. |
        | **Nominal Thickness (mm)** | Original nominal wall thickness of the pipe. |
        | **Minimum Thickness** | Minimum allowable thickness before repair or replacement. |
        | **Water Cut** | Percentage of water contained in the transported fluid. |
        | **OverallCR** | Overall Corrosion Rate in mm/year. |
        | **Soil pH** | Acidity or alkalinity level of the surrounding soil. |
        | **Design Pressure (psi)** | Maximum pressure for which the pipeline was designed. |
        | **Leak Count** | Number of leaks recorded in the pipeline's history. |
        | **Operating Temperature (F)** | Normal operating temperature of the fluid. |
        | **Insulation** | Presence of thermal insulation (YES/NO). |
        | **Pipe Type** | Pipe manufacturing process (ERW, SEAMLESS, SAWH, SAWL). |
        | **Pipe Position** | Location of the pipeline (Buried, Aboveground, River Crossing, etc.). |
        | **Fluid Rep** | Nature of the transported fluid (Gas, Heavy Oil, Water, etc.). |
        | **3oF** | Consequence of Failure category from A to E. |
        | **Soil Resistivity** | Soil's ability to resist electrical current (corrosivity indicator). |
        | **Flowrate (BFPD)** | Fluid flowrate in Barrels of Fluid Per Day. |
        | **Service_Age** | Number of years in service since installation. |
        | **HCA** | High Consequence Area indicator. |
        | **Internal CR** | Specific internal corrosion rate (mm/year). |
        | **PoF** | Probability of Failure category from 1 to 5. |
        | **H2S** | Hydrogen sulfide content (highly corrosive gas). |
        | **Coating** | Condition or type of external protective coating. |
        """)
    
    star_ph = " :red[*]" if phase_key == "Phase 1" else ""

    col1, col2, col3 = st.columns(3)
    
    with col1:
        nps = st.text_input("NPS (inch) :red[*]", "508.0")
        nom_t = st.text_input("Nominal Thickness (mm) :red[*]", "12.7")
        min_t = st.text_input("Minimum Thickness :red[*]", "9.5")
        wc = st.text_input("Water Cut :red[*]", "0.0")
        ocr = st.text_input("OverallCR :red[*]", "0.1")
        ph = st.text_input(f"Soil_pH{star_ph}", "7.0")
        pres = st.text_input("Design Pressure (psi) :red[*]", "1000.0")
        leak = st.text_input("Leak Count :red[*]", "0.0")
        
    with col2:
        loc_class = st.text_input("Operating Temperature (F) :red[*]", "200.0")
        # Transformation into selectbox for parameters containing options
        ins = st.selectbox("Insulation:red[*]", ["YES", "NO"], index=0)
        pt = st.selectbox("Pipe Type:red[*]", ["ERW", "SAWH", "SAWL", "SEAMLESS"], index=0)
        pp = st.selectbox("Pipe Position:red[*]", ["BURIED", "ABOVEGROUND", "RAWA", "RIVER CROSSING", "ROAD CROSSING"], index=1)
        fluid = st.selectbox("Fluid Rep :red[*]", ["GAS", "WATER", "CONDENSATE", "FOUL FLUID", "HEAVY OIL", "LIGHT OIL", "STEAM"], index=0)
        cof = st.selectbox("3oF:red[*]", ["A", "B", "C", "D", "E"], index=0)
        soil = st.selectbox("Soil Resistivity:red[*]", ["MILDLY", "MILDLY/MODERATELY", "MODERATELY", "VERY"], index=0)
        
    with col3:
        flow = st.text_input("Flowrate (BFPD) :red[*]", "416676.0")
        age = st.text_input("Service_Age", "73") # Never a star
        hca = st.text_input("HCA :red[*]", "1")
        internal = st.text_input("Internal CR :red[*]", "0.017119")
        # Transformation into selectbox for PoF and Coating
        pof = st.selectbox("PoF:red[*]", ["1", "2", "3", "4", "5"], index=0)
        h2s = st.text_input("H2S :red[*]", "1.0")
        coat = st.selectbox("Coating:red[*]", ["Poor/Unknown", "Bad", "Fair", "Good"], index=0)

    if st.button("💾 Load Manual Data"):
        data = {
            'NPS (inch)': nps, 'Nominal Thickness (mm)': nom_t, 'Minimum Thickness': min_t,
            'Insulation': ins, 'Water Cut': wc, 'OverallCR': ocr, 'Soil pH': ph,
            'Design Pressure (psi)': pres, 'Leak Count': leak, 'Location Class': loc_class,
            'Pipe Type': pt, 'Pipe Position': pp, 'Fluid Representative': fluid,
            '3oF': cof, 'Soil Resistivity': soil, 'Coating': coat,
            'Flowrate (BFPD)': flow, 'Service_Age': age, 'HCA': hca, 'Internal CR': internal,
            'PoF': pof, 'H2S': h2s
        }
        st.session_state.df_input = pd.DataFrame([data])
        st.success("Manual data loaded!")

# ============================================================================
# 4. EXECUTION (With highlighted results)
# ============================================================================
if not st.session_state.df_input.empty:
    df_input = st.session_state.df_input
    
    st.write("### 📊 Imported Data Characteristics")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(label="Total Rows", value=f"{df_input.shape[0]:,}")
    with c2:
        st.metric(label="Total Columns", value=df_input.shape[1])
    with c3:
        total_missing = int(df_input.isna().sum().sum())
        st.metric(label="Missing Values", value=total_missing, delta="+ OK" if total_missing == 0 else "⚠️ Check data")
        
    st.write("### 📋 Preview:")
    st.dataframe(df_input.head(3))
    
    if st.button("🚀 Run Diagnostic and Prediction", type="primary"):
        if assets is None:
            st.error(f"The required model file for {phase_key} is missing or could not be loaded on the server.")
        else:
            if "Unnamed: 0" in df_input.columns or "Étiquettes de lignes" in df_input.columns:
                st.error("🚨 Error: The uploaded file structure is invalid. It looks like a Pivot Table. Please upload a file containing raw tabular structured parameters.")
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
                            if 'Nominal_Thickness' in df_encoded.columns:
                                t_nom = pd.to_numeric(df_encoded['Nominal_Thickness'], errors='coerce').fillna(12.7)
                            else:
                                t_nom = pd.Series(12.7, index=df_encoded.index)
                                
                            if 'Minimum_Thickness' in df_encoded.columns:
                                t_min = pd.to_numeric(df_encoded['Minimum_Thickness'], errors='coerce').fillna(9.5)
                            else:
                                t_min = pd.Series(9.5, index=df_encoded.index)
                                
                            if 'NPS (inch)' in df_encoded.columns:
                                D = pd.to_numeric(df_encoded['NPS (inch)'], errors='coerce').fillna(508.0)
                            else:
                                D = pd.Series(508.0, index=df_encoded.index)
                                
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
                        
                        # Calculation of predictions
                        predictions = model.predict(scaler.transform(final_df))
                        rounded_preds = np.round(predictions, 2)
                        
                        # Sheet 1: Original Dataset + Results
                        df_result = df_input.copy()
                        df_result.insert(0, 'ESTIMATED LIFE (YEARS)', rounded_preds)
                        st.session_state.df_result = df_result
                        
                        # Sheet 2: Encoded Dataset + Results
                        df_encoded_result = final_df.copy()
                        df_encoded_result.insert(0, 'ESTIMATED LIFE (YEARS)', rounded_preds)
                        
                        # --- Creation of multi-sheet Excel export with Highlighting ---
                        excel_buffer = BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                            df_result.to_excel(writer, index=False, sheet_name='Original_Dataset_prediction')
                            df_encoded_result.to_excel(writer, index=False, sheet_name='Dataset_used_by_the_model')
                            
                            # Accessing Excel file objects
                            workbook = writer.book
                            worksheet1 = writer.sheets['Original_Dataset_prediction']
                            worksheet2 = writer.sheets['Dataset_used_by_the_model']
                            
                            # Creating highlight format (Yellow background, bold black text)
                            highlight_format = workbook.add_format({
                                'bg_color': '#FFC107',
                                'font_color': 'black',
                                'bold': True,
                                'border': 1
                            })
                            
                            # Applying highlight to column A (ESTIMATED LIFE)
                            worksheet1.set_column('A:A', 28, highlight_format)
                            worksheet2.set_column('A:A', 28, highlight_format)
                            
                        st.session_state.excel_data = excel_buffer.getvalue()
                        
                except Exception as e:
                    st.error(f"An error occurred during the calculation: {e}")

if st.session_state.df_result is not None:
    st.success("✅ Calculations completed successfully!")
    st.write("### 📈 Final Results:")
    
    # --- NEW: Highlighting the column on the Streamlit interface ---
    styled_result = st.session_state.df_result.style.set_properties(
        subset=['ESTIMATED LIFE (YEARS)'], 
        **{'background-color': '#ffc107', 'color': 'black', 'font-weight': 'bold'}
    )
    st.dataframe(styled_result)

    st.download_button(
        label="📥 Download Complete Results (Multi-Sheet Excel File)",
        data=st.session_state.excel_data,
        file_name=f"predictions_{phase_key.lower().replace(' ', '_')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary"
    )
