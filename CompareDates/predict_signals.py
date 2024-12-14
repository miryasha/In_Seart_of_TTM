import json
import pandas as pd
import numpy as np
import pickle

def predict_new_data(new_json_file):
    """
    Load trained model and predict signals for new data
    """
    # Load the saved model and components
    with open('signal_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('feature_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_columns.json', 'r') as f:
        feature_columns = json.load(f)
    
    # Read new JSON data
    with open(new_json_file, 'r') as f:
        price_data = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {
            'date': date,
            'open': float(values['1. open']),
            'high': float(values['2. high']),
            'low': float(values['3. low']),
            'close': float(values['4. close']),
            'volume': float(values['5. volume'])
        }
        for date, values in price_data['pricesInfo']['priceDataObj'].items()
    ])
    
    # Prepare features
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Calculate the same features used in training
    df['price_change'] = (df['close'] - df['open']) / df['open']
    df['high_low_range'] = (df['high'] - df['low']) / df['low']
    df['volume_change'] = df['volume'].pct_change()
    df['close_change'] = df['close'].pct_change()
    
    for i in range(1, 4):
        df[f'close_change_{i}'] = df['close_change'].shift(i)
        df[f'volume_change_{i}'] = df['volume_change'].shift(i)
    
    # Drop rows with NaN
    df = df.dropna()
    
    # Make predictions
    X = df[feature_columns]
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    
    # Create signals DataFrame
    signals = df[predictions != 'none'][['date']]
    signals['signal'] = predictions[predictions != 'none']
    
    # Generate output filename from input filename
    output_file = new_json_file.replace('.json', '_signals.js')
    
    # Save predictions
    with open(output_file, 'w') as f:
        f.write('[\n')
        for i, row in signals.iterrows():
            f.write(f"  {{ date: '{row['date'].strftime('%Y-%m-%d')}', signal: '{row['signal']}' }}")
            if i < len(signals) - 1:
                f.write(',\n')
            else:
                f.write('\n')
        f.write(']\n')
    
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predict_signals.py new_data.json")
        sys.exit(1)
    
    new_json_file = sys.argv[1]
    predict_new_data(new_json_file)