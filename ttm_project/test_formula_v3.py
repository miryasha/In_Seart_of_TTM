import json
import pandas as pd
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volume import VolumeWeightedAveragePrice
from ta.volatility import BollingerBands
import numpy as np

class SignalDetector:
    def __init__(self, pivot_lookback=2):
        self.pivot_lookback = pivot_lookback

    def calculate_pivot_points(self, df):
        """Calculate Pivot High and Pivot Low points"""
        # Initialize pivot columns
        df['pivot_high'] = np.nan
        df['pivot_low'] = np.nan
        
        for i in range(self.pivot_lookback, len(df) - self.pivot_lookback):
            window_high = df['high'].iloc[i-self.pivot_lookback:i+self.pivot_lookback+1]
            window_low = df['low'].iloc[i-self.pivot_lookback:i+self.pivot_lookback+1]
            
            if df['high'].iloc[i] == window_high.max():
                df.loc[df.index[i], 'pivot_high'] = df['high'].iloc[i]
            
            if df['low'].iloc[i] == window_low.min():
                df.loc[df.index[i], 'pivot_low'] = df['low'].iloc[i]
        
        df['last_pivot_high'] = df['pivot_high'].ffill()
        df['last_pivot_low'] = df['pivot_low'].ffill()
        
        df['last_pivot_high'] = df['last_pivot_high'].fillna(df['high'].max())
        df['last_pivot_low'] = df['last_pivot_low'].fillna(df['low'].min())
        
        df['distance_to_pivot_high'] = (df['last_pivot_high'] - df['close']) / df['close']
        df['distance_to_pivot_low'] = (df['close'] - df['last_pivot_low']) / df['close']
        
        df['pivot_range'] = df['last_pivot_high'] - df['last_pivot_low']
        df['pivot_range_pct'] = df['pivot_range'] / df['close']
        
        df['pivot_position'] = (df['close'] - df['last_pivot_low']) / df['pivot_range']
        
        return df
    
    def add_technical_indicators(self, df):
        """Add technical indicators"""
        # RSI
        df['rsi'] = RSIIndicator(close=df['close']).rsi()
        
        # Volume Analysis
        df['volume_change'] = df['volume'].pct_change()
        df['vwap'] = VolumeWeightedAveragePrice(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume']
        ).volume_weighted_average_price()
        
        # Trend strength
        df['trend_strength'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
        
        return df
    
    def detect_pattern(self, row):
        """Detect pattern based on current values"""
        try:
            pivot_position = row['pivot_position']
            rsi = row['rsi']
            distance_to_high = row['distance_to_pivot_high']
            distance_to_low = row['distance_to_pivot_low']
            trend_strength = row['trend_strength']
            
            # Buy conditions
            buy_signal = (
                (pivot_position < 0.40) and 
                (distance_to_low < 0.02) and 
                (rsi < 50) and 
                (distance_to_high > 0.03) and
                (trend_strength > 0.40)
            )
            
            # Sell conditions
            sell_signal = (
                (pivot_position > 0.70) and 
                (distance_to_high < 0.02) and 
                (rsi > 55) and 
                (distance_to_low > 0.04) and
                (trend_strength > 0.40)
            )
            
            if buy_signal:
                return 'buy'
            elif sell_signal:
                return 'sell'
            else:
                return 'none'
        except:
            return 'none'
    
    def process_data(self, json_file):
        """Process JSON data and detect signals"""
        # Load data
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
        df = df.sort_values('date').reset_index(drop=True)
        
        # Add indicators
        df = self.add_technical_indicators(df)
        df = self.calculate_pivot_points(df)
        
        # Fill NaN values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Detect patterns
        signals = []
        for idx, row in df.iterrows():
            signal = self.detect_pattern(row)
            if signal != 'none':
                signals.append({
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'signal': signal
                })
        
        return signals

def main():
    detector = SignalDetector()
    signals = detector.process_data('originalCall.json')
    
    # Save signals to JSON file
    with open('detected_signals.json', 'w') as f:
        json.dump(signals, f, indent=2)
    
    # Also save in the specific format for easy comparison
    with open('signals_for_comparison.js', 'w') as f:
        for signal in signals:
            f.write(f"{{ date: '{signal['date']}', signal: '{signal['signal']}' }},\n")
    
    print("\nSignals have been saved to:")
    print("1. detected_signals.json (JSON format)")
    print("2. signals_for_comparison.js (Comparison format)")

if __name__ == "__main__":
    main()