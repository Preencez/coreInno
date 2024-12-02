import streamlit as st
import numpy as np
import joblib

# Load the trained Gaussian Naive Bayes model
gaussian_nb_model = joblib.load("gaussian_nb_model.pkl")

# Symptom-to-number mapping
symptom_mapping = {
    "Abdominal discomfort": 1,
    "Blood in the semen": 2,
    "Blood in the urine": 3,
    "Blurry vision": 4,
    "Confusion": 5,
    "Cough": 6,
    "Cuts and bruises that are slow to heal": 7,
    "Dark colored urine": 8,
    "Diarrhea": 9,
    "Difficulty achieving or maintaining an erection": 10,
    "Difficulty starting or holding back urine": 11,
    "Discomfort on the right side of the abdomen below the ribs": 12,
    "Fatigue": 13,
    "Feeling very hungry": 14,
    "Feeling very thirsty": 15,
    "Feeling very tired": 16,
    "Fever": 17,
    "Frequent skin infection": 18,
    "Frequent urination especially at night": 19,
    "Headache": 20,
    "Joint pain": 21,
    "Longer periods than normal": 22,
    "Losing weight without trying": 23,
    "Loss of appetite": 24,
    "Loss of hunger": 25,
    "Loss of weight": 26,
    "Malaise": 27,
    "Mental difficulties": 28,
    "Muscle aches and joint pain": 29,
    "Nausea": 30,
    "Night sweats": 31,
    "Numbness or tingling in your hands or feet": 32,
    "Pain during sex": 33,
    "Painful ejaculation": 34,
    "Painful or burning urination": 35,
    "Pelvic pain": 36,
    "Periods that are heavier": 37,
    "Rash": 38,
    "Severe fatigue": 39,
    "Sore throat and painful mouth sores": 40,
    "Sores that heal slowly": 41,
    "Strong vaginal odor": 42,
    "Swelling in the abdomen": 43,
    "Swelling in the arms": 44,
    "Swelling in the legs": 45,
    "Swollen lymph glands on the neck": 46,
    "Tendency to bleed easily": 47,
    "Urinating frequently at night": 48,
    "Vaginal bleeding after menopause": 49,
    "Vaginal bleeding after sex": 50,
    "Vaginal bleeding between periods": 51,
    "Vaginal discharge that contains blood": 52,
    "Vaginal discharge that is watery": 53,
    "Very dry skin": 54,
    "Vomiting": 55,
    "Weak or interrupted flow of urine": 56,
    "Weakness or numbness in the legs or feet": 57,
    "Weight loss": 58,
    "Whites of the eyes": 59,
    "Yellowing of the skin": 60,
}

# Diseases and associated symptom codes
disease_symptoms = {
    "Hepatitis": [14, 1, 19, 12, 15, 17, 20, 16, 18, 13, 11, 11],
    "Liver Failure": [16, 21, 22, 23, 24, 25, 12, 14, 26, 27, 28, 29],
    "Diabetes": [16, 40, 35, 31, 30, 38, 39, 33, 32, 37, 34, 36],
    "Cervical Cancer": [48, 50, 47, 43, 44, 49, 46, 41, 45, 42, 11, 11],
    "HIV": [8, 7, 10, 5, 2, 6, 4, 9, 3, 1, 11, 11],
    "Prostate Cancer": [60, 59, 57, 55, 52, 53, 37, 54, 51, 61, 56, 58],
}

# Navigation
pages = ["Cover Page", "Prediction", "Disclaimer"]
selected_page = st.sidebar.radio("Navigate to", pages)

# Cover Page
if selected_page == "Cover Page":
    st.title("🌟 Welcome to the Disease Prediction System 🌟")
    st.image("lab_tech.webp", caption="Predict diseases with ease", use_container_width=True)

    st.write("""
        ### How to Use:
        1. Navigate to **Prediction** to input symptoms and predict diseases.
        2. Read the **Disclaimer** for important information.
    """)

# Prediction Page
elif selected_page == "Prediction":
    st.title("Disease Prediction System")

    # Display all symptoms in a straight line
    st.write("### Symptoms and Their Corresponding Numbers")
    symptom_lines = [f"{number}: {symptom}" for symptom, number in symptom_mapping.items()]
    st.write(", ".join(symptom_lines))  # Display symptoms in a single line

    # Display diseases and their associated symptom codes
    st.write("### Diseases and Associated Symptoms")
    for disease, symptoms in disease_symptoms.items():
        st.write(f"**{disease}**: {', '.join(map(str, symptoms))}")  # Display only numbers

    # Dropdown menus for selecting symptoms
    st.write("### Select Symptoms by Number")
    cols = st.columns(3)  # Divide dropdowns into 3 columns
    selected_symptoms = []

    for i in range(12):
        with cols[i % 3]:  # Distribute dropdowns across columns
            symptom_number = st.number_input(
                f"Symptom {i+1} (Enter Number)", min_value=0, max_value=60, step=1, key=f"symptom_number_{i+1}"
            )
            if symptom_number != 0:
                selected_symptoms.append(symptom_number)

    st.write("### Selected Symptoms")
    st.write(", ".join(map(str, selected_symptoms)) if selected_symptoms else "No symptoms selected.")

    # Prediction function
    def manual_prediction(symptom_values):
        input_array = np.array(symptom_values).reshape(1, -1)
        return gaussian_nb_model.predict(input_array)[0]

    # Predict Button
    if st.button("Make Prediction"):
        if len(selected_symptoms) < 3:
            st.error("Please select at least 3 symptoms for prediction.")
        else:
            prediction = manual_prediction(selected_symptoms)
            st.success(f"The predicted disease is: **{prediction}**")
        st.warning("For the most accurate prediction, please input all associated symptoms.")

    # Disclaimer under Prediction Page
    st.markdown("""
    ---
    **Disclaimer:** This system provides predictions based on selected symptoms but is not a substitute for professional medical advice. Please consult a licensed physician for further evaluation.
    """)

# Disclaimer Page
elif selected_page == "Disclaimer":
    st.title("Disclaimer")
    st.write("""
        ### Important Information:
        - This tool is not a substitute for professional medical advice, diagnosis, or treatment.
        - Always consult a physician or qualified healthcare provider with any health-related questions or concerns.
        - The predictions provided are based on a trained machine learning model and may not always be accurate.
        - By using this system, you agree to take full responsibility for consulting a professional for any medical conditions.
    """)
