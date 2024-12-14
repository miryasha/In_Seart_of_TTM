import json
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

def predict_signals(json_file):
    # Load model and scaler
    model = tf.keras.models.load_model('signal_model')
    with open('feature_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_columns.json', 'r') as f:
        feature_columns = json.load(f)
    
    # Read and prepare new data
    with open(json_file, 'r') as f:
        price_data = json.load(f)
    
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
    
    # Calculate same features as training data
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    df['price_change'] = (df['close'] - df['open']) / df['open']
    df['high_low_range'] = (df['high'] - df['low']) / df['low']
    df['volume_change'] = df['volume'].pct_change()
    df['close_change'] = df['close'].pct_change()
    
    for i in range(1, 4):
        df[f'close_change_{i}'] = df['close_change'].shift(i)
        df[f'volume_change_{i}'] = df['volume_change'].shift(i)
        df[f'high_low_range_{i}'] = df['high_low_range'].shift(i)
    
    for period in [5, 10, 20]:
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
    
    # Remove rows with NaN
    df = df.dropna()
    
    # Prepare features for prediction
    X = df[feature_columns].values
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    signals = np.argmax(predictions, axis=1)
    
    # Convert predictions to signals
    signal_map = {0: 'none', 1: 'buy', -1: 'sell'}
    df['predicted_signal'] = signals
    df['predicted_signal'] = df['predicted_signal'].map({0: 'none', 1: 'buy', 2: 'sell'})
    
    # Create output
    signals = df[df['predicted_signal'].isin(['buy', 'sell'])][['date', 'predicted_signal']]
    
    # Save predictions
    with open('predicted_signals.js', 'w') as f:
        f.write('[\n')
        for i, (_, row) in enumerate(signals.iterrows()):
            f.write(f"  {{ date: '{row['date'].strftime('%Y-%m-%d')}', signal: '{row['predicted_signal']}' }}")
            if i < len(signals) - 1:
                f.write(',\n')
            else:
                f.write('\n')
        f.write(']\n')
    
    print(f"Predictions saved to predicted_signals.js")

if __name__ == "__main__":
    input_file = "new_data.json"  # Your new data file
    predict_signals(input_file)