# predict.py
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import joblib
from datetime import datetime

def load_model_and_scaler():
    try:
        model = joblib.load('price_prediction_model.joblib')
        scaler = joblib.load('price_prediction_scaler.joblib')
        with open('model_config.txt', 'r') as f:
            # Read until we find the Feature Columns section
            feature_columns = []
            for line in f:
                if line.strip() == "Feature Columns:":
                    break
            for line in f:
                if line.strip() == "":  # Stop at empty line
                    break
                feature_columns.append(line.strip())
        return model, scaler, feature_columns
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

def get_data_for_date_range(start_date, end_date=None):
    print(f"Retrieving data from {start_date} to {end_date if end_date else 'latest'}...")
    engine = create_engine('mysql+pymysql://root:rootroot@localhost/money')
    
    # Modified query to include lag data
    if end_date:
        query = text("""
            WITH numbered_rows AS (
                SELECT 
                    *,
                    LAG(C_W, 1) OVER (ORDER BY starting_date) as prev_C_W,
                    LAG(C_W, 2) OVER (ORDER BY starting_date) as prev_2_C_W
                FROM no_price_anaysis_tbl
            )
            SELECT *
            FROM numbered_rows
            WHERE starting_date >= :start_date
            AND starting_date <= :end_date
            ORDER BY starting_date;
        """)
        params = {"start_date": start_date, "end_date": end_date}
    else:
        query = text("""
            WITH numbered_rows AS (
                SELECT 
                    *,
                    LAG(C_W, 1) OVER (ORDER BY starting_date) as prev_C_W,
                    LAG(C_W, 2) OVER (ORDER BY starting_date) as prev_2_C_W
                FROM no_price_anaysis_tbl
            )
            SELECT *
            FROM numbered_rows
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

def create_advanced_features(df):
    """Create the same advanced features as in training"""
    
    # Fill NaN values in lagged features
    df['prev_C_W'] = df['prev_C_W'].fillna(0)
    df['prev_2_C_W'] = df['prev_2_C_W'].fillna(0)
    
    # Create trend features
    df['weekly_trend'] = np.where(df['C_sum_ground'] > 0, 1, 
                                 np.where(df['C_sum_ground'] < 0, -1, 0))
    
    # Create momentum features
    df['momentum'] = df['C_W'] * abs(df['C_sum_ground'])
    
    # Create volatility feature
    df['day_changes'] = df[['C_1', 'C_2', 'C_3', 'C_4', 'C_5']].nunique(axis=1)
    
    # Create pattern features
    df['reversal_pattern'] = np.where(
        (df['C_sum_ground'].abs() >= 3) & (df['C_W'] != df['prev_C_W']), 1, 0
    )
    
    # Create consistency features
    df['direction_consistency'] = df[['C_1', 'C_2', 'C_3', 'C_4', 'C_5']].apply(
        lambda x: x.value_counts().max() / 5, axis=1
    )
    
    # Ensure no NaN values remain
    df = df.fillna(0)
    
    return df

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
    
    # Create advanced features
    df = create_advanced_features(df)
    
    # Prepare features
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
        'prob_neutral': probabilities[:, 1] if probabilities.shape[1] > 2 else 0,
        'prob_up': probabilities[:, -1]
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