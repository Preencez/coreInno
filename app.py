import streamlit as st
import numpy as np
import joblib
import pandas as pd

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

    # Create a table with six columns
    st.write("### Symptom Numbers and Their Descriptions")
    symptom_list = list(symptom_mapping.items())  # Convert the mapping to a list of tuples
    num_columns = 6  # Number of columns for the table
    rows = len(symptom_list) // num_columns + (1 if len(symptom_list) % num_columns != 0 else 0)

    # Prepare table data for six columns
    table_data = [
        [(symptom_list[i + rows * j][1], symptom_list[i + rows * j][0]) if i + rows * j < len(symptom_list) else ("", "") for j in range(num_columns)]
        for i in range(rows)
    ]

    # Format the data into "Number: Symptom" and handle empty cells
    formatted_data = [
        ["{}: {}".format(number, symptom) if number else "" for number, symptom in row]
        for row in table_data
    ]

    # Convert the formatted data into a DataFrame for display
    col_df = pd.DataFrame(formatted_data, columns=[f"Column {i+1}" for i in range(num_columns)])
    st.table(col_df)

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
