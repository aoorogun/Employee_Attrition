import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from joblib import dump, load

def load_data(data_path):
    data = pd.read_csv(data_path)
    columns_to_drop = ['BusinessTravel', 'DailyRate', 'Department', 'EducationField', 'EmployeeCount',
                       'EmployeeNumber', 'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',
                       'JobSatisfaction', 'MonthlyRate', 'Over18', 'OverTime', 'PercentSalaryHike',
                       'PerformanceRating', 'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
                       'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsSinceLastPromotion']
    data.drop(columns=columns_to_drop, inplace=True)
    X = data.drop('Attrition', axis=1)
    y = data['Attrition']
    return X, y, data

def create_preprocessor(X):
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'bool']).columns
    
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ])
    
    return preprocessor

def train_and_save_models(X, y, output_dir='saved_models'):
    preprocessor = create_preprocessor(X)
    
    models = {
        'Logistic_Regression': LogisticRegression(max_iter=1000),
        'Decision_Tree': DecisionTreeClassifier(),
        'Random_Forest': RandomForestClassifier(),
        'SVM': SVC(probability=True)
    }
    
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        pipeline.fit(X, y)
        model_path = f"{output_dir}/{name}.joblib"
        dump(pipeline, model_path)
        
    print("All models have been trained and saved.")

def make_prediction(model_name, user_input, model_dir='saved_models'):
    model_path = f"{model_dir}/{model_name}.joblib"
    model = load(model_path)
    prediction = model.predict(pd.DataFrame([user_input]))
    return prediction[0]

if __name__ == "__main__":
    X, y = load_data('path/to/your_dataset.csv')
    train_and_save_models(X, y)
