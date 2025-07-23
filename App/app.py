import streamlit as st
import pandas as pd 
import joblib

model=joblib.load("model.pkl")
scaler=joblib.load("scaler.pkl")
expected_columns=joblib.load("columns.pkl")

st.title("Heart Stroke Prediction ❤️")
st.markdown("Provide The Following Deatils")

age=st.slider("Age",18,100,40)
sex=st.selectbox('SEX',['M','F'])
chest_pain=st.selectbox('Chest Pain',['TA','ATA','NAP','ASY'])
resting_bp=st.number_input("Resting Blood Pressure (mm HG)",80,200,120)
cholesterol=st.number_input("Cholesterol (mg/dl)",100,600,200)
fasting_bs=st.selectbox("Fasting Boold Suger >120 mm/dl",[0,1])
resting_ecg=st.selectbox("Resting ECG",['LVH','Normal','ST'])
max_hr=st.slider("Max Heart Rate",60,220,150)
excersise_angina=st.selectbox("Excersise Induced Angina",["Y","N"])
oldpeak=st.slider("Oldpeak (ST Depression)",0.0,6.2,1.0)
st_slop=st.selectbox("ST Slop",['Up','Flat','Down'])

if st.button('Predict'):
    raw_input = {
        "Age": age,
        "sex": sex,
        "ChestPain": chest_pain,
        "RestingBp": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBs": fasting_bs,
        "RestingEcg": resting_ecg,
        "MaxHr": max_hr,
        "ExcersiseAngina": excersise_angina,
        "Oldpeak": oldpeak,
        "StSlop": st_slop
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([raw_input])

    # Make sure all expected columns are present
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0  # Default fill

    # Reorder columns to match training set
    input_df = input_df[expected_columns]

    # Predict using raw input (NO SCALING)
    prediction = model.predict(input_df)[0]

    # Show result
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    # Show raw input
    st.subheader("Raw Input")
    st.write(input_df)