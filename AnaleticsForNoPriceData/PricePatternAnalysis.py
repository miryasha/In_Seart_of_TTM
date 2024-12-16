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
    
    # Open file for writing results
    with open('price_analysis_results.txt', 'w') as f:
        # Write basic statistics
        f.write("=== Basic Statistics ===\n")
        f.write(df.describe().to_string())
        f.write("\n\n")
        
        # Correlation analysis - exclude date and id columns
        numeric_columns = ['C_1', 'C_2', 'C_3', 'C_4', 'C_5', 'C_W', 
                          'C_sum_positive', 'C_sum_negetive', 'C_sum_ground', 
                          'C_multiply', 'next_C_W']
        
        f.write("=== Correlation Analysis ===\n")
        correlation_matrix = df[numeric_columns].corr()
        f.write("\nCorrelation with next_C_W:\n")
        f.write(correlation_matrix['next_C_W'].sort_values(ascending=False).to_string())
        f.write("\n\n")
        
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
        
        # Model performance
        f.write("=== Model Performance ===\n")
        f.write(f"\nAccuracy: {accuracy_score(y_test, y_pred)}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred))
        f.write("\n")
        
        # Feature importance analysis
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        f.write("=== Feature Importance ===\n")
        f.write(feature_importance.to_string())
        f.write("\n\n")
        
        # Pattern analysis
        f.write("=== Pattern Analysis ===\n")
        
        # Analyze success rate based on previous C_W
        f.write("\nPrediction success rate based on previous C_W:\n")
        for prev_cw in df['C_W'].unique():
            next_cw = df[df['C_W'] == prev_cw]['next_C_W']
            if len(next_cw) > 0:
                f.write(f"\nWhen C_W is {prev_cw}:\n")
                f.write(next_cw.value_counts(normalize=True).to_string())
                f.write("\n")
        
        # Analyze patterns based on sum_ground
        f.write("\nNext C_W distribution based on C_sum_ground:\n")
        for ground in df['C_sum_ground'].unique():
            next_cw = df[df['C_sum_ground'] == ground]['next_C_W']
            if len(next_cw) > 0:
                f.write(f"\nWhen C_sum_ground is {ground}:\n")
                f.write(next_cw.value_counts(normalize=True).to_string())
                f.write("\n")
        
        # Write model parameters for future use
        f.write("\n=== Model Parameters for Prediction ===\n")
        f.write("Feature order for prediction input:\n")
        for feature in feature_columns:
            f.write(f"- {feature}\n")
            
        # Write some prediction guidelines
        f.write("\nPrediction Guidelines:\n")
        f.write("1. Most influential features (top 3):\n")
        for idx, row in feature_importance.head(3).iterrows():
            f.write(f"   - {row['feature']}: {row['importance']:.4f}\n")
            
        f.write("\n2. Strong correlations with next_C_W (|correlation| > 0.1):\n")
        strong_corr = correlation_matrix['next_C_W'][abs(correlation_matrix['next_C_W']) > 0.1]
        for feature, corr in strong_corr.items():
            if feature != 'next_C_W':
                f.write(f"   - {feature}: {corr:.4f}\n")
                
        # Save example predictions
        f.write("\nExample Predictions:\n")
        example_cases = [
            {'case': 'All Positive', 'values': {'C_1': 1, 'C_2': 1, 'C_3': 1, 'C_4': 1, 'C_5': 1,
                                              'C_W': 1, 'C_sum_positive': 5, 'C_sum_negetive': 0,
                                              'C_sum_ground': 5, 'C_multiply': 0}},
            {'case': 'All Negative', 'values': {'C_1': -1, 'C_2': -1, 'C_3': -1, 'C_4': -1, 'C_5': -1,
                                               'C_W': -1, 'C_sum_positive': 0, 'C_sum_negetive': -5,
                                               'C_sum_ground': -5, 'C_multiply': 0}},
            {'case': 'Mixed Case', 'values': {'C_1': 1, 'C_2': -1, 'C_3': 1, 'C_4': -1, 'C_5': 1,
                                            'C_W': 1, 'C_sum_positive': 3, 'C_sum_negetive': -2,
                                            'C_sum_ground': 1, 'C_multiply': 6}}
        ]
        
        for case in example_cases:
            prediction = predict_next_cw(rf_model, scaler, case['values'])
            f.write(f"\n{case['case']}:")
            f.write(f"\nPredicted next_C_W: {prediction['prediction']}")
            f.write(f"\nProbabilities: {prediction['probabilities']}\n")
    
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
    print("Analysis complete. Results written to 'price_analysis_results.txt' and 'correlation_heatmap.png'")