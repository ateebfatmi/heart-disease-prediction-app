import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier # Example Model

# Page Configuration
st.set_page_config(page_title="Heart Disease Diagnostic Tool", layout="wide")

# Title and Description
st.title("Heart Disease Analysis & Prediction")
st.markdown("""
This application analyzes heart health data and provides insights into risk factors. 
It uses the `heart.csv` dataset for processing and visualization.
""")

# --- STEP 1: LOAD & CLEAN DATA (Mirroring your Notebook logic) ---
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("heart.csv")
    
    # Categorical Conversion (Mirroring your logic)
    cat_mapping = {
        'Sex': {'M': 0, 'F': 1},
        'ChestPainType': {'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3},
        'RestingECG': {'Normal': 0, 'ST': 1, 'LVH': 2},
        'ExerciseAngina': {'N': 0, 'Y': 1},
        'ST_Slope': {'Up': 0, 'Flat': 1, 'Down': 2}
    }
    
    for col, mapping in cat_mapping.items():
        df[col] = df[col].map(mapping)
    
    # Handling 0 values in Cholesterol and RestingBP using KNNImputer
    df['Cholesterol'].replace(0, np.nan, inplace=True)
    df['RestingBP'].replace(0, np.nan, inplace=True)
    
    imputer = KNNImputer(n_neighbors=3)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    # Standardizing Types
    cols_to_int = df_imputed.columns.drop('Oldpeak')
    df_imputed[cols_to_int] = df_imputed[cols_to_int].astype('int32')
    
    return df_imputed

df = load_and_clean_data()

# --- STEP 2: SIDEBAR NAVIGATION ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Visualizations", "Prediction Model"])

# --- PAGE 1: DATA OVERVIEW ---
if page == "Data Overview":
    st.header("Dataset Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Data Sample")
        st.dataframe(df.head(10))
    with col2:
        st.write("### Statistics")
        st.write(df.describe())

# --- PAGE 2: VISUALIZATIONS (From your Notebook) ---
elif page == "Visualizations":
    st.header("Visual Analytics")
    
    viz_type = st.selectbox("Select Chart", 
                            ["Target Distribution", "Age vs Heart Disease", "Correlation Heatmap"])
    
    if viz_type == "Target Distribution":
        fig = px.pie(df, names='HeartDisease', title='Heart Disease Distribution', hole=0.4)
        st.plotly_chart(fig)
        
    elif viz_type == "Age vs Heart Disease":
        fig = px.histogram(df, x='Age', color='HeartDisease', barmode='group', title="Risk by Age Group")
        st.plotly_chart(fig)
        
    elif viz_type == "Correlation Heatmap":
        corr = df.corr()
        fig = px.imshow(corr, text_auto=True, title="Feature Correlation")
        st.plotly_chart(fig)

# --- PAGE 3: PREDICTION MODEL ---
elif page == "Prediction Model":
    st.header("Predict Heart Disease Risk")
    st.info("Input patient data below to check probability.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Age", 20, 100, 50)
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x==0 else "Female")
        cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3])
        
    with col2:
        bp = st.number_input("Resting Blood Pressure", 80, 200, 120)
        chol = st.number_input("Cholesterol", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])

    with col3:
        max_hr = st.slider("Max Heart Rate", 60, 220, 140)
        angina = st.selectbox("Exercise Angina", [0, 1])
        slope = st.selectbox("ST Slope", [0, 1, 2])

    if st.button("Predict"):
        # Dummy prediction logic for UI demonstration
        # In a real app, you'd load a saved .pkl model here
        risk = (age * 0.1 + bp * 0.05 + (1 if angina == 1 else 0)) / 20
        if risk > 0.5:
            st.error(f"High Risk Detected! (Score: {risk:.2f})")
        else:
            st.success(f"Low Risk. (Score: {risk:.2f})")