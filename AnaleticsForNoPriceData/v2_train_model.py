# train_model.py
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

def get_training_data():
    print("Connecting to database...")
    engine = create_engine('mysql+pymysql://root:rootroot@localhost/money')
    
    query = """
    WITH numbered_rows AS (
        SELECT 
            *,
            LEAD(C_W) OVER (ORDER BY starting_date) as next_C_W
        FROM no_price_anaysis_tbl
        WHERE starting_date <= '2024-01-01'  # Training cutoff date
    )
    SELECT * FROM numbered_rows
    WHERE next_C_W IS NOT NULL;
    """
    
    try:
        df = pd.read_sql(query, engine)
        print(f"Retrieved {len(df)} rows for training")
        return df
    except Exception as e:
        print(f"Error retrieving data: {e}")
        return None
    finally:
        engine.dispose()

def train_and_save_model():
    # Get data
    df = get_training_data()
    if df is None:
        return
    
    # Prepare features
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
    
    # Train model
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Save model and scaler
    print("Saving model and scaler...")
    joblib.dump(model, 'price_prediction_model.joblib')
    joblib.dump(scaler, 'price_prediction_scaler.joblib')
    
    # Save feature columns order
    with open('feature_columns.txt', 'w') as f:
        f.write('\n'.join(feature_columns))
    
    print("Model and scaler saved successfully")
    
    return model, scaler, feature_columns

if __name__ == "__main__":
    train_and_save_model()