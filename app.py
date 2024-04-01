import streamlit as st
import pandas as pd
from prediction import make_prediction

# Load available models from the prediction.py script
models = {name: f"{name.replace(' ', '_')}.joblib" for name in ["Logistic Regression", "Decision Tree", "Random Forest", "SVM"]}

st.title("Employee Attrition Prediction App")

# Get user input for each feature
user_input = {}
categorical_features = ["BusinessTravel", "Department", "Education", "EducationField", "Gender", "JobRole", 
                        "MaritalStatus", "Over18", "OverTime", "RelationshipSatisfaction", "WorkLifeBalance"]

for feature in ["Age", "DailyRate", "DistanceFromHome", "EmployeeCount", "EnvironmentSatisfaction", 
                "HourlyRate", "JobInvolvement", "JobLevel", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked", 
                "PercentSalaryHike", "PerformanceRating", "StandardHours", "StockOptionLevel", "TotalWorkingYears", 
                "TrainingTimesLastYear", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"]:
    if feature in categorical_features:
        user_input[feature] = st.selectbox(feature, list(pd.read_csv("path_to_your_dataset.csv")[feature].unique()))  # Load unique values from dataset
    else:
        user_input[feature] = st.text_input(feature)

# Allow user to select the model
selected_model = st.selectbox("Select Model", list(models.keys()))

# Button to make prediction
if st.button("Predict"):
    # Convert user input to DataFrame
    user_data_df = pd.DataFrame([user_input])

    # Make prediction using the chosen model
    prediction = make_prediction(models[selected_model], user_data_df)
    prediction_text = "Employee is predicted to " if prediction == 0 else "Employee is predicted NOT to "
    st.write(f"{prediction_text}leave the company.")
