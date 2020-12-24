import numpy as np
import pickle
import pandas as pd
import streamlit as st

with open('model_ml.pkl', 'rb') as file:
    classifier = pickle.load(file)
# @app.route('/')
def welcome():
    return "Welcome All"


# @app.route('/predict',methods=["Get"])
def predict_note_authentication(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca,
                                thal):
    X = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    prediction = classifier.predict(X.astype(np.float))
    print(prediction)
    return prediction


def main():
    st.title("Heart Diseases Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Heart Disease ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    age = st.text_input("Age", "Type Here")
    sex = st.text_input("Sex", "Type Here")
    cp = st.text_input("Chest Pain Type", "Type Here")
    trestbps = st.text_input("Resting Blood Pressure", "Type Here")
    chol = st.text_input("Cholestrol", "Type Here")
    fbs = st.text_input("Fasting Blood Sugar", "Type Here")
    restecg = st.text_input("Resting ECG", "Type Here")
    thalach = st.text_input("Max Heart Rate achieved", "Type Here")
    exang = st.text_input("Exercise Induced Angina", "Type Here")
    oldpeak = st.text_input("ST depression", "Type Here")
    slope = st.text_input("Slope", "Type Here")
    ca = st.text_input("Number of Major Vessels", "Type Here")
    thal = st.text_input("Thalium Stress Result", "Type Here")
    result = ""
    if st.button("Predict"):
        result = predict_note_authentication(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca,
                               thal)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")


if __name__ == '__main__':
    main()





