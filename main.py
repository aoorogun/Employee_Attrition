import streamlit as st
import pandas as pd
from prediction import load_data, train_and_save_models, make_prediction

X, y, data = load_data('Employee-Attrition (1).csv')  # Adjust the path as necessary
st.title("Employee Attrition Prediction")
user_input_data = {}
columns_per_row = 3
total_columns = len(X.columns)
columns = st.columns(columns_per_row)

for index, column in enumerate(X.columns):
    current_col_index = index % columns_per_row
    current_column = columns[current_col_index]
    if X[column].dtype == 'object':
        options = list(X[column].unique()) 
        selected = current_column.selectbox(f"Select {column}", options=options, index=0, key=column)
        user_input_data[column] = selected
    else:
        default_value = int(X[column].mean())
        user_input = current_column.number_input(f"Input {column}", value=default_value, key=column)
        user_input_data[column] = user_input
user_input_df = pd.DataFrame([user_input_data])
user_input_df = user_input_df.reindex(columns=X.columns)
model_names = ['Logistic_Regression', 'Decision_Tree', 'Random_Forest', 'SVM']
selected_model_name = st.selectbox("Select Model for Prediction", options=model_names)
if st.button("Predict"):
    prediction = make_prediction(selected_model_name, user_input_df)
    prediction_text = 'Likely to leave' if prediction == 1 else 'Likely to stay'
    st.write(f"Prediction with {selected_model_name}: {prediction_text}")
