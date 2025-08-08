import streamlit as st
import pandas as pd
import joblib

# Load  trained model
model = joblib.load("student_dropout_predict.pkl")  
features = [
    'Age', 'Gender', 'Study_Time', 'Number_of_Failures',
    'Parental_Status', 'Mother_Education', 'Father_Education',
    'Extra_Paid_Class', 'Wants_Higher_Education', 'Internet_Access',
    'Number_of_Absences', 'Final_Grade'
]
st.title("🎓 Student Dropout Risk Predictor")

st.markdown("Fill in the student's details below to predict dropout risk:")

# Input form
age = st.number_input("Student's Age", min_value=15, max_value=25, value=17)
gender = st.selectbox("Gender", ['F', 'M'])
study_time = st.slider("Weekly Study Time (1=low, 4=high)", 1, 4, 2)
failures = st.slider("Number of Past Class Failures", 0, 3, 0)
family_support = st.radio("Family Support", ['yes', 'no'])
internet = st.radio("Internet Access at Home", ['yes', 'no'])
absences = st.number_input("Number of School Absences", min_value=0, max_value=100, value=5)
higher_edu = st.radio("Wants Higher Education", ['yes', 'no'])
alcohol = st.slider("Weekend Alcohol Consumption (1=low, 5=high)", 1, 5, 2)

# Map input
user_input = {
    'Age': age,
    'Gender': 0 if gender == 'F' else 1,
    'Study_Time': study_time,
    'Number_of_Failures': failures,
    'Parental_Status': 1,  # Placeholder: you can replace with real input if needed
    'Mother_Education': 2,  # Placeholder
    'Father_Education': 2,  # Placeholder
    'Extra_Paid_Class': 0,  # Placeholder
    'Wants_Higher_Education': 1 if higher_edu == 'yes' else 0,
    'Internet_Access': 1 if internet == 'yes' else 0,
    'Number_of_Absences': absences,
    'Final_Grade': 10  # Placeholder
}


# user_df = pd.DataFrame([user_input])[model.features]
# user_df = pd.DataFrame([user_input], columns=features)
user_df = pd.DataFrame([user_input], columns=features)

# Predict button
if st.button("🔍 Predict Dropout Risk"):
    prediction = model.predict(user_df)[0]
    prediction_proba = model.predict_proba(user_df)[0]

    st.subheader("📊 Prediction Result:")
    if prediction == 1:
        st.error("⚠️ Risk of Dropping Out")
        st.write(f"**Probability:** {prediction_proba[1]*100:.2f}%")
    else:
        st.success("✅ Likely to Continue")
        st.write(f"**Probability:** {prediction_proba[0]*100:.2f}%")

    # Reasoning
    st.markdown("### 🧠 Reasoning")
    if prediction == 1:
        if failures > 0:
            st.write(f"- {failures} past failures")
        if absences > 5:
            st.write(f"- High absences ({absences})")
        if alcohol >= 4:
            st.write("- High weekend alcohol use")
        if family_support == 'no':
            st.write("- No family support")
        if higher_edu == 'no':
            st.write("- No interest in higher education")
    else:
        if failures == 0:
            st.write("- No past failures")
        if alcohol <= 2:
            st.write("- Low alcohol consumption")
        if higher_edu == 'yes':
            st.write("- Motivated for higher education")
