# predict.py
import pandas as pd
from sqlalchemy import create_engine
import joblib

def load_model_and_scaler():
    try:
        model = joblib.load('price_prediction_model.joblib')
        scaler = joblib.load('price_prediction_scaler.joblib')
        with open('feature_columns.txt', 'r') as f:
            feature_columns = f.read().splitlines()
        return model, scaler, feature_columns
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

def get_latest_data():
    print("Retrieving latest data...")
    engine = create_engine('mysql+pymysql://root:rootroot@localhost/money')
    
    query = """
    SELECT *
    FROM no_price_anaysis_tbl
    WHERE starting_date > '2024-01-01'  # Test data cutoff date
    ORDER BY starting_date DESC
    LIMIT 1;
    """
    
    try:
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        print(f"Error retrieving data: {e}")
        return None
    finally:
        engine.dispose()

def predict_next_week():
    # Load model and scaler
    model, scaler, feature_columns = load_model_and_scaler()
    if model is None:
        return
    
    # Get latest data
    df = get_latest_data()
    if df is None or len(df) == 0:
        print("No data available for prediction")
        return
    
    # Prepare features
    latest_data = df[feature_columns]
    
    # Scale features
    scaled_features = scaler.transform(latest_data)
    
    # Make prediction
    prediction = model.predict(scaled_features)
    probabilities = model.predict_proba(scaled_features)
    
    # Prepare results
    result = {
        'date': df['starting_date'].iloc[0],
        'prediction': prediction[0],
        'probabilities': {
            '-1': probabilities[0][0],
            '0': probabilities[0][1],
            '1': probabilities[0][2]
        }
    }
    
    # Print results
    print("\nPrediction Results:")
    print(f"Date: {result['date']}")
    print(f"Predicted next C_W: {result['prediction']}")
    print("Probabilities:")
    for k, v in result['probabilities'].items():
        print(f"  Class {k}: {v:.4f}")
    
    return result

if __name__ == "__main__":
    predict_next_week()