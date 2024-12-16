# predict.py
import pandas as pd
from sqlalchemy import create_engine, text
import joblib
from datetime import datetime

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

def get_data_for_date_range(start_date, end_date=None):
    print(f"Retrieving data from {start_date} to {end_date if end_date else 'latest'}...")
    engine = create_engine('mysql+pymysql://root:rootroot@localhost/money')
    
    if end_date:
        query = text("""
            SELECT *
            FROM no_price_anaysis_tbl
            WHERE starting_date >= :start_date
            AND starting_date <= :end_date
            ORDER BY starting_date;
        """)
        params = {"start_date": start_date, "end_date": end_date}
    else:
        query = text("""
            SELECT *
            FROM no_price_anaysis_tbl
            WHERE starting_date >= :start_date
            ORDER BY starting_date;
        """)
        params = {"start_date": start_date}
    
    try:
        df = pd.read_sql(query, engine, params=params)
        print(f"Retrieved {len(df)} rows of data")
        return df
    except Exception as e:
        print(f"Error retrieving data: {e}")
        return None
    finally:
        engine.dispose()

def predict_for_period(start_date, end_date=None):
    # Load model and scaler
    model, scaler, feature_columns = load_model_and_scaler()
    if model is None:
        return
    
    # Get data for specified period
    df = get_data_for_date_range(start_date, end_date)
    if df is None or len(df) == 0:
        print("No data available for prediction")
        return
    
    # Prepare features and make predictions
    features = df[feature_columns]
    scaled_features = scaler.transform(features)
    predictions = model.predict(scaled_features)
    probabilities = model.predict_proba(scaled_features)
    
    # Create results dataframe
    results = pd.DataFrame({
        'date': df['starting_date'],
        'actual_C_W': df['C_W'],
        'predicted_next_C_W': predictions,
        'prob_down': probabilities[:, 0],
        'prob_neutral': probabilities[:, 1],
        'prob_up': probabilities[:, 2]
    })
    
    # Print detailed results
    print("\nPrediction Results:")
    print("\nSummary:")
    print(f"Total predictions: {len(results)}")
    print(f"Predicted Up (1): {(predictions == 1).sum()}")
    print(f"Predicted Down (-1): {(predictions == -1).sum()}")
    print(f"Predicted Neutral (0): {(predictions == 0).sum()}")
    
    print("\nDetailed Predictions:")
    for _, row in results.iterrows():
        print(f"\nDate: {row['date'].strftime('%Y-%m-%d')}")
        print(f"Actual C_W: {row['actual_C_W']}")
        print(f"Predicted next C_W: {row['predicted_next_C_W']}")
        print("Probabilities:")
        print(f"  Down  (-1): {row['prob_down']:.4f}")
        print(f"  Neutral (0): {row['prob_neutral']:.4f}")
        print(f"  Up     (1): {row['prob_up']:.4f}")
    
    # Save results to CSV
    filename = f'predictions_{start_date}_to_{end_date if end_date else "latest"}.csv'
    results.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")
    
    return results

if __name__ == "__main__":
    print("Price Prediction Tool")
    print("1. Predict for a single date")
    print("2. Predict for a date range")
    choice = input("Enter your choice (1 or 2): ")
    
    try:
        if choice == "1":
            date = input("Enter date (YYYY-MM-DD): ")
            results = predict_for_period(date)
        elif choice == "2":
            start_date = input("Enter start date (YYYY-MM-DD): ")
            end_date = input("Enter end date (YYYY-MM-DD): ")
            results = predict_for_period(start_date, end_date)
        else:
            print("Invalid choice")
    except Exception as e:
        print(f"Error occurred: {e}")