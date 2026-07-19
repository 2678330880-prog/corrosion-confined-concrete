import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Global Webpage Settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="Concrete Constitutive Prediction System", layout="wide", page_icon="🏗️")
st.title("🧮 Stress-strain model prediction system for circular-section corroded stirrup-confined concrete")
st.markdown("Supports single-component parameter calculation and Excel/CSV batch prediction. Automatically outputs **peak stress**, **peak strain**, **deformation coefficient (r)**, and the **constitutive curve**.")

st.info("**Theoretical Basis**: The constitutive curve of this system is based on the **Mander Concrete Constitutive Model**. The equation is as follows:")
st.latex(r"f_c = \frac{f_{cc}' \cdot x \cdot r}{r - 1 + x^r} \quad (x = \varepsilon / \varepsilon_{cc})")

st.markdown("---")

# 2. Cache and Load Models
@st.cache_resource
def load_models():
    try:
        stress_path = os.path.join(BASE_DIR, "my_stress_model.pkl")
        strain_path = os.path.join(BASE_DIR, "my_strain_model.pkl")
        r_path = os.path.join(BASE_DIR, "my_r_model.pkl")
        
        if not all(os.path.exists(p) for p in [stress_path, strain_path, r_path]):
            return None, None, None
            
        model_stress = joblib.load(stress_path)   
        model_strain = joblib.load(strain_path)
        model_r = joblib.load(r_path)
        return model_stress, model_strain, model_r
        
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None, None

model_stress, model_strain, model_r = load_models()

if model_stress is None:
    st.error("Model files not found! Please ensure `my_stress_model.pkl` and the other two files are in the same directory as this script.")
    st.stop()

# ==========================================
# Feature Mapping Dictionary (Crucial for Model Compatibility)
# Maps Professional English UI names back to the Model's Chinese training names
# ==========================================
FEATURE_MAP_ENG_TO_CHN = {
    'Stirrup Corrosion Rate (%)': '箍筋锈蚀率(%)',
    'Longitudinal Bar Corrosion Rate (%)': '纵筋锈蚀率(%)',
    'Unconfined Concrete Strength (Axial)': '混凝土强度(轴心)',
#    'Specimen Width-to-Thickness Ratio': '试件宽厚比',
    'Stirrup Yield Strength': '箍筋屈服强度',
    'Longitudinal Bar Yield Strength': '纵筋屈服强度',
    'Volumetric Stirrup Ratio (%)': '体积配箍率(%)',
    'Longitudinal Reinforcement Ratio (%)': '纵筋配筋率(%)',
    'Concrete Peak Strain': '混凝土峰值应变',
    'Stirrup Spacing (mm)': '箍筋间距(mm)',
    'Stirrup Characteristic Value': '配箍特征值',
    'Longitudinal Reinforcement Characteristic Value': '配纵筋特征值'
}

# ==========================================
# Left Control Panel: Mode Selection
# ==========================================
st.sidebar.header("⚙️ Select Work Mode")
mode = st.sidebar.radio("Please select:", ["🖱️ Single Sample Manual Prediction", "📁 Batch Table Upload Prediction"])
st.sidebar.markdown("---")

# Fully localized to Professional English
REQUIRED_COLS = [
    'Stirrup Corrosion Rate (%)', 'Longitudinal Bar Corrosion Rate (%)', 
    'Unconfined Concrete Strength (Axial)'  # , 'Specimen Width-to-Thickness Ratio', 
    'Stirrup Yield Strength', 'Longitudinal Bar Yield Strength', 
    'Volumetric Stirrup Ratio (%)', 'Longitudinal Reinforcement Ratio (%)', 
    'Concrete Peak Strain', 'Stirrup Spacing (mm)'
]

if mode == "🖱️ Single Sample Manual Prediction":
    st.sidebar.header("📝 Input Component Parameters")

    st.sidebar.subheader("General Parameters")
    corrosion_stirrup = st.sidebar.number_input("Stirrup Corrosion Rate (%)", value=6.98, format="%g", step=0.1, min_value=0.0)
    corrosion_long = st.sidebar.number_input("Longitudinal Bar Corrosion Rate (%)", value=3.35, format="%g", step=0.1, min_value=0.0)
    fc = st.sidebar.number_input("Unconfined Concrete Strength (Axial) (MPa)", value=29.56, format="%g", step=0.1, min_value=0.0)
#    ratio_wh = st.sidebar.number_input("Specimen Width-to-Thickness Ratio", value=1.0, format="%g", step=0.1, min_value=0.0)

    st.sidebar.subheader("Reinforcement & Strength Parameters")
    fy_stirrup = st.sidebar.number_input("Stirrup Yield Strength (MPa)", value=324.0, format="%g", step=5.0, min_value=0.0)
    fy_long = st.sidebar.number_input("Longitudinal Bar Yield Strength (MPa)", value=475.0, format="%g", step=5.0, min_value=0.0)
    rho_v = st.sidebar.number_input("Volumetric Stirrup Ratio (%)", value=1.52, format="%g", step=0.1, min_value=0.0)
    rho_l = st.sidebar.number_input("Longitudinal Reinforcement Ratio (%)", value=2.67, format="%g", step=0.1, min_value=0.0)
    fy_stirrup = fy_stirrup * (1 - 1.695756 * corrosion_stirrup / 100)
    fy_long = fy_long * (1 - 1.695756 * corrosion_long / 100)
    rho_v = rho_v * (100 - corrosion_stirrup) * 0.01
    rho_l = rho_l * (100 - corrosion_long) * 0.01

    st.sidebar.subheader("Peak Strain Parameters")
    strain_peak = st.sidebar.number_input("Unconfined Concrete Peak Strain", value=0.002, format="%g", step=0.0001, min_value=0.0)

    st.sidebar.subheader("Deformation Characteristic Parameters")
    spacing_stirrup = st.sidebar.number_input("Stirrup Spacing (mm)", value=60.0, format="%g", step=5.0, min_value=0.0)
    
    char_stirrup = rho_v * 0.01 * fy_stirrup / fc
    char_long = rho_l * 0.01 * fy_long / fc

    st.subheader("Single Sample Prediction & Constitutive Analysis")
    if st.button("Start Prediction"):
        # UI DataFrame completely in English
        df_stress = pd.DataFrame({
            'Stirrup Corrosion Rate (%)': [corrosion_stirrup], 'Longitudinal Bar Corrosion Rate (%)': [corrosion_long],
            'Stirrup Yield Strength': [fy_stirrup], 'Longitudinal Bar Yield Strength': [fy_long],
            'Volumetric Stirrup Ratio (%)': [rho_v], 'Longitudinal Reinforcement Ratio (%)': [rho_l],
            'Unconfined Concrete Strength (Axial)': [fc]  # , 'Specimen Width-to-Thickness Ratio': [ratio_wh]
        })
        
        df_strain = pd.DataFrame({
            'Stirrup Corrosion Rate (%)': [corrosion_stirrup], 'Longitudinal Bar Corrosion Rate (%)': [corrosion_long],
            'Stirrup Yield Strength': [fy_stirrup], 'Longitudinal Bar Yield Strength': [fy_long],
            'Volumetric Stirrup Ratio (%)': [rho_v], 'Longitudinal Reinforcement Ratio (%)': [rho_l],
            'Unconfined Concrete Strength (Axial)': [fc], 'Concrete Peak Strain': [strain_peak]  # , 'Specimen Width-to-Thickness Ratio': [ratio_wh]
        })
        
        df_r = pd.DataFrame({
            'Stirrup Corrosion Rate (%)': [corrosion_stirrup], 'Longitudinal Bar Corrosion Rate (%)': [corrosion_long],
            'Stirrup Spacing (mm)': [spacing_stirrup], 'Stirrup Characteristic Value': [char_stirrup],
            'Longitudinal Reinforcement Characteristic Value': [char_long], 'Unconfined Concrete Strength (Axial)': [fc]  # , 'Specimen Width-to-Thickness Ratio': [ratio_wh]
        })
        
        # Rename English columns to Chinese under the hood before predicting
        pred_stress = model_stress.predict(df_stress.rename(columns=FEATURE_MAP_ENG_TO_CHN).values)[0]
        pred_strain = model_strain.predict(df_strain.rename(columns=FEATURE_MAP_ENG_TO_CHN).values)[0] / 10000.0
        pred_r = model_r.predict(df_r.rename(columns=FEATURE_MAP_ENG_TO_CHN).values)[0]
        
        st.success("Calculation Complete!")
        
        res_col_left, res_col_right = st.columns([1, 2])
        
        with res_col_left:
            st.markdown("#### 📊 Prediction Metrics")
            st.metric(label="Peak Stress (MPa)", value=f"{pred_stress:.2f}")
            st.metric(label="Peak Strain", value=f"{pred_strain:.4f}") 
            st.metric(label="Deformation Coefficient (r)", value=f"{pred_r:.2f}")
                
        with res_col_right:
            st.markdown("#### 📈 Constitutive Skeleton Curve (Stress-Strain)")
            
            peak_stress = pred_stress
            peak_strain_val = pred_strain 
            r_val = max(pred_r, 1.05) 
            
            eps_array = np.linspace(0, peak_strain_val * 3, 200)
            x_array = eps_array / peak_strain_val
            
            stress_array = (peak_stress * x_array * r_val) / (r_val - 1 + x_array**r_val)
            
            chart_df = pd.DataFrame({
                'Strain': eps_array,
                'Stress': stress_array
            })
            
            st.line_chart(
                chart_df, 
                x='Strain', 
                y='Stress', 
                x_label="Strain ε", 
                y_label="Stress fc (MPa)"
            )

# ==========================================
# Batch Table Prediction Logic
# ==========================================
elif mode == "📁 Batch Table Upload Prediction":
    st.subheader("Batch Prediction")
    
    st.info(f"💡 Please upload a table (CSV/Excel) containing the following **10 basic feature columns**:\n\n"
            f"`{'`, `'.join(REQUIRED_COLS)}`")
    
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                try:
                    df_input = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df_input = pd.read_csv(uploaded_file, encoding='gbk') 
            else:
                df_input = pd.read_excel(uploaded_file)
                
            st.write("Preview of uploaded data:")
            st.dataframe(df_input.head())
            
            missing_cols = [col for col in REQUIRED_COLS if col not in df_input.columns]
            if missing_cols:
                st.error(f"❌ Failed: The table is missing the following required columns: **{', '.join(missing_cols)}**. Please ensure the header names match exactly.")
            else:
                if st.button("🚀 Start Batch Prediction"):
                    with st.spinner("Calculating rapidly..."):
                        df_work = df_input.copy()
                        
                        # Calculate features using English column names
                        df_work['Stirrup Characteristic Value'] = df_work['Volumetric Stirrup Ratio (%)'] * 0.01 * df_work['Stirrup Yield Strength'] / df_work['Unconfined Concrete Strength (Axial)']
                        df_work['Longitudinal Reinforcement Characteristic Value'] = df_work['Longitudinal Reinforcement Ratio (%)'] * 0.01 * df_work['Longitudinal Bar Yield Strength'] / df_work['Unconfined Concrete Strength (Axial)']
                        
                        df_stress = df_work[['Stirrup Corrosion Rate (%)', 'Longitudinal Bar Corrosion Rate (%)', 'Stirrup Yield Strength', 'Longitudinal Bar Yield Strength', 'Volumetric Stirrup Ratio (%)', 'Longitudinal Reinforcement Ratio (%)', 'Unconfined Concrete Strength (Axial)']]  # , 'Specimen Width-to-Thickness Ratio'
                        df_strain = df_work[['Stirrup Corrosion Rate (%)', 'Longitudinal Bar Corrosion Rate (%)', 'Stirrup Yield Strength', 'Longitudinal Bar Yield Strength', 'Volumetric Stirrup Ratio (%)', 'Longitudinal Reinforcement Ratio (%)', 'Unconfined Concrete Strength (Axial)', 'Concrete Peak Strain']]  # , 'Specimen Width-to-Thickness Ratio'
                        df_r = df_work[['Stirrup Corrosion Rate (%)', 'Longitudinal Bar Corrosion Rate (%)', 'Stirrup Spacing (mm)', 'Stirrup Characteristic Value', 'Longitudinal Reinforcement Characteristic Value', 'Unconfined Concrete Strength (Axial)']]  # , 'Specimen Width-to-Thickness Ratio'
                        
                        # Rename English columns to Chinese under the hood before predicting
                        df_input['Predicted_Peak_Stress (MPa)'] = model_stress.predict(df_stress.rename(columns=FEATURE_MAP_ENG_TO_CHN).values)
                        df_input['Predicted_Peak_Strain'] = model_strain.predict(df_strain.rename(columns=FEATURE_MAP_ENG_TO_CHN).values) / 10000.0
                        df_input['Predicted_Deformation_Coefficient (r)'] = model_r.predict(df_r.rename(columns=FEATURE_MAP_ENG_TO_CHN).values)
                        
                        st.success(f"🎉 Successfully completed predictions for {len(df_input)} rows of data!")
                        st.dataframe(df_input)
                        
                        csv_data = df_input.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label="📥 Download Complete Prediction Results Table",
                            data=csv_data,
                            file_name="Batch_Prediction_Results.csv",
                            mime="text/csv",
                            type="primary" 
                        )
                        
        except Exception as e:
            st.error(f"❌ An unknown error occurred while reading the file or predicting. Please ensure the data format is correct. Details: {e}")