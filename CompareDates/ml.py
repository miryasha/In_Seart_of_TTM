import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from datetime import datetime
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
    
    # Add lagged features (previous N days)
    for i in range(1, 4):  # Previous 3 days
        df[f'close_change_{i}'] = df['close_change'].shift(i)
        df[f'volume_change_{i}'] = df['volume_change'].shift(i)
        df[f'high_low_range_{i}'] = df['high_low_range'].shift(i)
    
    # Calculate moving averages
    for period in [5, 10, 20]:
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
    
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
    
    # Convert signals to numeric
    signal_map = {'none': 0, 'buy': 1, 'sell': -1}
    df['signal_numeric'] = df['signal'].map(signal_map)
    
    # Remove rows with NaN values
    df = df.dropna()
    
    return df

def create_and_train_model(df):
    # Prepare features and target
    feature_columns = [
        'price_change', 'high_low_range', 'volume_change', 'close_change',
        'close_change_1', 'close_change_2', 'close_change_3',
        'volume_change_1', 'volume_change_2', 'volume_change_3',
        'high_low_range_1', 'high_low_range_2', 'high_low_range_3',
        'sma_5', 'sma_10', 'sma_20',
        'volume_sma_5', 'volume_sma_10', 'volume_sma_20'
    ]
    
    X = df[feature_columns].values
    y = pd.get_dummies(df['signal'])  # One-hot encode signals
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Create model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(len(feature_columns),)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(3, activation='softmax')  # 3 classes: none, buy, sell
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Save model and scaler
    model.save('signal_model')
    with open('feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature columns
    with open('feature_columns.json', 'w') as f:
        json.dump(feature_columns, f)
    
    return model, scaler, feature_columns

if __name__ == "__main__":
    # Prepare data
    df = prepare_data('originalCall.json', 'JustOutPut.js')
    
    # Train model
    model, scaler, feature_columns = create_and_train_model(df)
    
    print("Model trained and saved as 'signal_model'")
    print("Scaler saved as 'feature_scaler.pkl'")
    print("Feature columns saved as 'feature_columns.json'")