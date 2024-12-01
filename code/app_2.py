import streamlit as st
import requests
import pandas as pd


def main():
    st.title("Heart Disease Prediction")

    # Tạo form nhập liệu
    age = st.number_input("Age", 1, 100)
    gender = st.selectbox("Gender", [1, 0])
    chestpain = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    restingBP = st.number_input("Resting Blood Pressure", 94, 200)
    serumcholestrol = st.number_input("Serum Cholestrol", 126, 564)
    fastingbloodsugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restingrelectro = st.selectbox("Resting Electrocardiogram", [0, 1, 2])
    maxheartrate = st.number_input("Maximum Heart Rate Achieved", 71, 202)
    exerciseangia = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("Oldpeak", 0.0, 6.2)
    slope = st.selectbox("Slope of ST Segment", [1, 2, 3])
    noofmajorvessels = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])

    if st.button('Predict'):
        # Prepare input data
        input_data = {
            'age': age,
            'gender': gender,
            'chestpain': chestpain,
            'restingBP': restingBP,
            'serumcholestrol': serumcholestrol,
            'fastingbloodsugar': fastingbloodsugar,
            'restingrelectro': restingrelectro,
            'maxheartrate': maxheartrate,
            'exerciseangia': exerciseangia,
            'oldpeak': oldpeak,
            'slope': slope,
            'noofmajorvessels': noofmajorvessels
        }

        # Call the FastAPI endpoint
        response = requests.post("http://127.0.0.1:8000/predict", json=input_data)

        if response.status_code == 200:
            result = response.json()
            prediction = result["prediction"]
            confidence = result["confidence"]

            bg_color = 'red' if prediction == 'Positive' else 'green'
            st.markdown(
                f"<p style='background-color:{bg_color}; color:white; padding:10px;'>"
                f"Prediction: {prediction}<br>Confidence: {confidence}%</p>",
                unsafe_allow_html=True
            )
        else:
            st.error("Error occurred during prediction.")


if __name__ == '__main__':
    main()
