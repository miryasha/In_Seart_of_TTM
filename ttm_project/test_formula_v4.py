import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD, CCIIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
import json

class SignalDetector:
    def __init__(self):
        # Thresholds based on the analysis results
        self.buy_conditions = {
            'rsi': {'value': 46.03, 'weight': 0.2},
            'stoch_k': {'value': 42.36, 'weight': 0.15},
            'cci': {'value': -37.71, 'weight': 0.15},
            'macd_diff': {'value': -0.63, 'weight': 0.15},
            'trend_strength': {'value': 0.47, 'weight': 0.15},
            'pivot_position': {'value': 0.37, 'weight': 0.2}
        }
        
        self.sell_conditions = {
            'rsi': {'value': 55.98, 'weight': 0.2},
            'stoch_k': {'value': 72.26, 'weight': 0.15},
            'cci': {'value': 80.31, 'weight': 0.15},
            'macd_diff': {'value': 0.77, 'weight': 0.15},
            'trend_strength': {'value': 0.48, 'weight': 0.15},
            'pivot_position': {'value': 0.77, 'weight': 0.2}
        }
        
        self.confidence_threshold = 0.3

    def prepare_data(self, json_file):
        """Prepare data with technical indicators."""
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
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Calculate technical indicators
        df['rsi'] = RSIIndicator(close=df['close']).rsi()
        
        stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
        df['stoch_k'] = stoch.stoch()
        
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        df['cci'] = CCIIndicator(high=df['high'], low=df['low'], close=df['close']).cci()
        
        # Calculate trend strength and pivot position
        df['trend_strength'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
        
        # Calculate pivot points and position
        df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
        df['pivot_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        return df.fillna(0)

    def calculate_signal_confidence(self, row, conditions):
        """Calculate confidence score for a signal."""
        confidence = 0
        
        for indicator, params in conditions.items():
            if indicator in row:
                if indicator == 'rsi':
                    if (conditions == self.buy_conditions and row[indicator] < params['value']) or \
                       (conditions == self.sell_conditions and row[indicator] > params['value']):
                        confidence += params['weight']
                
                elif indicator == 'stoch_k':
                    if (conditions == self.buy_conditions and row[indicator] < params['value']) or \
                       (conditions == self.sell_conditions and row[indicator] > params['value']):
                        confidence += params['weight']
                
                elif indicator == 'cci':
                    if (conditions == self.buy_conditions and row[indicator] < params['value']) or \
                       (conditions == self.sell_conditions and row[indicator] > params['value']):
                        confidence += params['weight']
                
                elif indicator == 'macd_diff':
                    if (conditions == self.buy_conditions and row[indicator] > params['value']) or \
                       (conditions == self.sell_conditions and row[indicator] < params['value']):
                        confidence += params['weight']
                
                elif indicator in ['trend_strength', 'pivot_position']:
                    if abs(row[indicator] - params['value']) < 0.1:
                        confidence += params['weight']
        
        return confidence

    def detect_signals(self, df):
        """Detect trading signals based on technical patterns."""
        signals = []
        
        for idx, row in df.iterrows():
            if idx < 10:  # Skip first few rows for indicator calculation
                continue
            
            buy_confidence = self.calculate_signal_confidence(row, self.buy_conditions)
            sell_confidence = self.calculate_signal_confidence(row, self.sell_conditions)
            
            signal = None
            confidence = 0
            
            if buy_confidence > sell_confidence and buy_confidence >= self.confidence_threshold:
                signal = 'buy'
                confidence = buy_confidence
            elif sell_confidence > buy_confidence and sell_confidence >= self.confidence_threshold:
                signal = 'sell'
                confidence = sell_confidence
            
            if signal:
                signals.append({
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'signal': signal,
                    'confidence': round(confidence, 4),
                    'price': round(row['close'], 4),
                    'indicators': {
                        'rsi': round(row['rsi'], 2),
                        'stoch_k': round(row['stoch_k'], 2),
                        'cci': round(row['cci'], 2),
                        'macd_diff': round(row['macd_diff'], 4),
                        'trend_strength': round(row['trend_strength'], 4),
                        'pivot_position': round(row['pivot_position'], 4)
                    }
                })
        
        return signals

    def process_data(self, json_file):
        """Process data and generate signals."""
        print("Loading and preparing data...")
        df = self.prepare_data(json_file)
        
        print("Detecting signals...")
        signals = self.detect_signals(df)
        
        print(f"Found {len(signals)} signals")
        return signals


def main():
    detector = SignalDetector()
    signals = detector.process_data('originalCall.json')
    
    # Save signals to JSON file
    with open('detected_signals.json', 'w') as f:
        json.dump(signals, f, indent=2)
    
    # Save in comparison format
    with open('signals_for_comparison.js', 'w') as f:
        for signal in signals:
            f.write(f"{{ date: '{signal['date']}', signal: '{signal['signal']}' }},\n")
    
    print("\nSignals have been saved to:")
    print("1. detected_signals.json (JSON format with full details)")
    print("2. signals_for_comparison.js (Simple format for comparison)")
    
    # Print summary statistics
    buy_signals = len([s for s in signals if s['signal'] == 'buy'])
    sell_signals = len([s for s in signals if s['signal'] == 'sell'])
    
    print("\nSignal Summary:")
    print(f"Total Signals: {len(signals)}")
    print(f"Buy Signals: {buy_signals}")
    print(f"Sell Signals: {sell_signals}")
    
    avg_confidence = sum(s['confidence'] for s in signals) / len(signals)
    print(f"Average Signal Confidence: {avg_confidence:.4f}")


if __name__ == "__main__":
    main()