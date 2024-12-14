import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import json
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volume import VolumeWeightedAveragePrice
from ta.volatility import BollingerBands

class AdvancedPatternFinder:
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.model = None
        
    def prepare_data(self, json_file, signal_file):
        # Load price data
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
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Add technical indicators
        self.add_technical_indicators(df)
        
        # Load signals
        with open(signal_file, 'r') as f:
            content = f.read()
            import re
            pattern = r"{[\s]*date:[\s]*'([^']+)'[\s]*,[\s]*signal:[\s]*'([^']+)'[\s]*}"
            signals = re.findall(pattern, content)
        
        signal_df = pd.DataFrame(signals, columns=['date', 'signal'])
        signal_df['date'] = pd.to_datetime(signal_df['date'])
        
        # Merge signals
        df['signal'] = 'none'
        for idx, row in signal_df.iterrows():
            df.loc[df['date'] == row['date'], 'signal'] = row['signal']
        
        return df
    
    def add_technical_indicators(self, df):
        """Add various technical indicators"""
        # Price Changes
        df['price_change'] = df['close'].pct_change()
        df['high_low_range'] = (df['high'] - df['low']) / df['low']
        
        # Moving Averages
        for period in [5, 8, 13, 21]:
            df[f'ema_{period}'] = EMAIndicator(close=df['close'], window=period).ema_indicator()
        
        # MACD
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # RSI
        df['rsi'] = RSIIndicator(close=df['close']).rsi()
        
        # Bollinger Bands
        bb = BollingerBands(close=df['close'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        
        # Volume Analysis
        df['volume_change'] = df['volume'].pct_change()
        df['vwap'] = VolumeWeightedAveragePrice(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume']
        ).volume_weighted_average_price()
        
        # Pattern Detection (Fixed version)
        df['high_1d_change'] = df['high'].pct_change()
        df['high_2d_change'] = df['high'].pct_change(periods=2)
        df['high_3d_change'] = df['high'].pct_change(periods=3)
        
        df['low_1d_change'] = df['low'].pct_change()
        df['low_2d_change'] = df['low'].pct_change(periods=2)
        df['low_3d_change'] = df['low'].pct_change(periods=3)
        
        # Trend strength
        df['trend_strength'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
        
        return df
    
    def create_sequences(self, df):
        """Create sequences for LSTM"""
        feature_cols = [col for col in df.columns if col not in ['date', 'signal']]
        X, y = [], []
        
        for i in range(len(df) - self.sequence_length):
            sequence = df[feature_cols].iloc[i:(i + self.sequence_length)].values
            label = df['signal'].iloc[i + self.sequence_length]
            X.append(sequence)
            y.append(label)
        
        X = np.array(X)
        y = pd.get_dummies(y)  # One-hot encode the signals
        
        return X, y
    
    def build_model(self, input_shape, output_shape):
        """Build LSTM model"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(output_shape, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def analyze_patterns(self, df):
        """Analyze patterns in the data"""
        patterns = {signal_type: {} for signal_type in ['buy', 'sell']}
        
        for signal_type in patterns.keys():
            signal_data = df[df['signal'] == signal_type]
            
            if not signal_data.empty:
                for col in df.columns:
                    if col not in ['date', 'signal'] and not signal_data[col].isnull().all():
                        values = signal_data[col].dropna()
                        if not values.empty:
                            patterns[signal_type][col] = {
                                'mean': values.mean(),
                                'std': values.std(),
                                'min': values.min(),
                                'max': values.max()
                            }
        
        return patterns
    
    def find_patterns(self, json_file, signal_file):
        """Main method to find patterns"""
        print("Loading and preparing data...")
        df = self.prepare_data(json_file, signal_file)
        
        print("\nAnalyzing technical patterns...")
        patterns = self.analyze_patterns(df)
        
        print("\nPreparing sequences for LSTM...")
        X, y = self.create_sequences(df)
        
        print("\nTraining LSTM model...")
        self.model = self.build_model(
            input_shape=(X.shape[1], X.shape[2]),
            output_shape=y.shape[1]
        )
        
        history = self.model.fit(
            X, y,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        # Save results
        self.save_findings(patterns, df)
        
        return patterns, history
    
    def save_findings(self, patterns, df):
        """Save the discovered patterns and conditions"""
        with open('advanced_patterns.txt', 'w') as f:
            f.write("Advanced Pattern Analysis\n\n")
            
            for signal_type in ['buy', 'sell']:
                f.write(f"\n{signal_type.upper()} Patterns:\n")
                if patterns[signal_type]:
                    for indicator, stats in patterns[signal_type].items():
                        if abs(stats['mean']) > stats['std']:
                            f.write(f"\n{indicator}:\n")
                            f.write(f"  Mean: {stats['mean']:.4f}\n")
                            f.write(f"  Range: {stats['min']:.4f} to {stats['max']:.4f}\n")
                            f.write(f"  Typical range: {stats['mean'] - stats['std']:.4f} to {stats['mean'] + stats['std']:.4f}\n")
                
            f.write("\nSequence Patterns:\n")
            f.write("- Analysis based on last 10 price bars\n")
            f.write("- Considers technical indicators and their relationships\n")
            f.write("- Uses LSTM for complex pattern detection\n")

if __name__ == "__main__":
    finder = AdvancedPatternFinder()
    patterns, history = finder.find_patterns('originalCall.json', 'JustOutPut.js')
    
    print("\nPattern analysis complete! Check advanced_patterns.txt for results.")