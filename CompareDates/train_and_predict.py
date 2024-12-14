import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

def prepare_data(json_file, signal_file):
    # Read price data
    with open(json_file, 'r') as f:
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
    
    # Convert date and sort
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Calculate technical features
    df['price_change'] = (df['close'] - df['open']) / df['open']
    df['high_low_range'] = (df['high'] - df['low']) / df['low']
    df['volume_change'] = df['volume'].pct_change()
    df['close_change'] = df['close'].pct_change()
    
    # Add lagged features
    for i in range(1, 4):
        df[f'close_change_{i}'] = df['close_change'].shift(i)
        df[f'volume_change_{i}'] = df['volume_change'].shift(i)
    
    # Read signal data
    with open(signal_file, 'r') as f:
        content = f.read()
        import re
        pattern = r"{[\s]*date:[\s]*'([^']+)'[\s]*,[\s]*signal:[\s]*'([^']+)'[\s]*}"
        signals = re.findall(pattern, content)
    
    # Create signals DataFrame
    signals_df = pd.DataFrame(signals, columns=['date', 'signal'])
    signals_df['date'] = pd.to_datetime(signals_df['date'])
    
    # Add signals to main DataFrame
    df['signal'] = 'none'
    for idx, row in signals_df.iterrows():
        df.loc[df['date'] == row['date'], 'signal'] = row['signal']
    
    # Drop rows with NaN
    df = df.dropna()
    
    return df

def train_model(df):
    # Prepare features
    feature_columns = [
        'price_change', 'high_low_range', 'volume_change', 'close_change',
        'close_change_1', 'close_change_2', 'close_change_3',
        'volume_change_1', 'volume_change_2', 'volume_change_3'
    ]
    
    X = df[feature_columns]
    y = df['signal']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    # Save model and scaler
    with open('signal_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('feature_columns.json', 'w') as f:
        json.dump(feature_columns, f)
    
    return model, scaler, feature_columns

def predict_signals(model, scaler, feature_columns, json_file, output_file):
    # Read new data
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
    
    # Prepare features
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
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

if __name__ == "__main__":
    # Train the model
    print("Preparing data...")
    df = prepare_data('originalCall.json', 'JustOutPut.js')
    
    print("Training model...")
    model, scaler, feature_columns = train_model(df)
    
    # Make predictions on the same data to verify
    print("Generating predictions...")
    predict_signals(model, scaler, feature_columns, 'originalCall.json', 'predicted_signals_v2.js')
    
    print("Done! Check predicted_signals.js for results.")