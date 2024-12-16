# First, install SQLAlchemy:
# pip install sqlalchemy pymysql

from sqlalchemy import create_engine
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.inspection import permutation_importance
import seaborn as sns
import matplotlib.pyplot as plt

def get_data():
    print("Attempting to connect to database...")
    engine = create_engine('mysql+pymysql://root:rootroot@localhost/money')
    
    # Simplified query using ROW_NUMBER() instead of subquery
    query = """
    WITH numbered_rows AS (
        SELECT 
            *,
            LEAD(C_W) OVER (ORDER BY starting_date) as next_C_W
        FROM no_price_anaysis_tbl
    )
    SELECT * FROM numbered_rows
    WHERE next_C_W IS NOT NULL;
    """
    
    try:
        print("Executing query...")
        df = pd.read_sql(query, engine)
        print(f"Retrieved {len(df)} rows of data")
        if len(df) == 0:
            print("Warning: No data was retrieved!")
        return df
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None
    finally:
        print("Closing database connection...")
        engine.dispose()

def analyze_patterns():
    # Get data
    df = get_data()
    # if df is None or len(df) == 0:
    #     print("No data available for analysis. Please check your database connection and data.")
    #     return None, None

    # Basic statistical analysis
    print("\n=== Basic Statistics ===")
    print(df.describe())
    
    # Correlation analysis - exclude date and id columns
    numeric_columns = ['C_1', 'C_2', 'C_3', 'C_4', 'C_5', 'C_W', 
                      'C_sum_positive', 'C_sum_negetive', 'C_sum_ground', 
                      'C_multiply', 'next_C_W']
    
    print("\n=== Correlation Analysis ===")
    correlation_matrix = df[numeric_columns].corr()
    print("\nCorrelation with next_C_W:")
    print(correlation_matrix['next_C_W'].sort_values(ascending=False))
    
    # Create correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()
    
    # Prepare data for prediction
    feature_columns = ['C_1', 'C_2', 'C_3', 'C_4', 'C_5', 'C_W', 
                      'C_sum_positive', 'C_sum_negetive', 'C_sum_ground', 'C_multiply']
    X = df[feature_columns]
    y = df['next_C_W']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test_scaled)
    
    # Print model performance
    print("\n=== Model Performance ===")
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== Feature Importance ===")
    print(feature_importance)
    
    # Pattern analysis
    print("\n=== Pattern Analysis ===")
    
    # Analyze success rate based on previous C_W
    print("\nPrediction success rate based on previous C_W:")
    for prev_cw in df['C_W'].unique():
        next_cw = df[df['C_W'] == prev_cw]['next_C_W']
        if len(next_cw) > 0:
            print(f"When C_W is {prev_cw}:")
            print(next_cw.value_counts(normalize=True))
    
    # Analyze patterns based on sum_ground
    print("\nNext C_W distribution based on C_sum_ground:")
    for ground in df['C_sum_ground'].unique():
        next_cw = df[df['C_sum_ground'] == ground]['next_C_W']
        if len(next_cw) > 0:
            print(f"\nWhen C_sum_ground is {ground}:")
            print(next_cw.value_counts(normalize=True))
    
    return rf_model, scaler

def predict_next_cw(model, scaler, current_values):
    """
    Predict next C_W based on current values
    
    current_values should be a dictionary with all required features
    """
    features = pd.DataFrame([current_values])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)
    
    return {
        'prediction': prediction[0],
        'probabilities': {i: prob for i, prob in enumerate(probabilities[0])}
    }

if __name__ == "__main__":
    print("Starting analysis...")
    # Train model and analyze patterns
    model, scaler = analyze_patterns()
    
    if model is None or scaler is None:
        print("Could not complete analysis due to data issues.")
        exit()
        
    # Example prediction
    example_values = {
        'C_1': 1, 'C_2': -1, 'C_3': 1, 'C_4': 0, 'C_5': 1,
        'C_W': 1, 'C_sum_positive': 3, 'C_sum_negetive': -1,
        'C_sum_ground': 2, 'C_multiply': 3
    }
    
    prediction = predict_next_cw(model, scaler, example_values)
    print("\n=== Example Prediction ===")
    print(f"Predicted next C_W: {prediction['prediction']}")
    print("Prediction probabilities:", prediction['probabilities'])