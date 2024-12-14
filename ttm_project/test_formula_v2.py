import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
import json

class TTMSignalTester:
    def __init__(self, rsi_buy_threshold=50, rsi_sell_threshold=55,
                 volatility_threshold=0.015, volume_multiplier=1.2,
                 trend_threshold=0.01):
        self.rsi_buy_threshold = rsi_buy_threshold
        self.rsi_sell_threshold = rsi_sell_threshold
        self.volatility_threshold = volatility_threshold
        self.volume_multiplier = volume_multiplier
        self.trend_threshold = trend_threshold
        
    def prepare_data(self, json_file):
        """Load and prepare price data"""
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
        df = df.sort_values('date').set_index('date')
        
        return df
    
    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        # RSI
        df['rsi'] = RSIIndicator(close=df['close']).rsi()
        
        # Volatility
        df['high_low_range'] = (df['high'] - df['low']) / df['low']
        
        # Price changes
        df['high_3d_change'] = df['high'].pct_change(periods=3)
        df['low_3d_change'] = df['low'].pct_change(periods=3)
        
        # Volume
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        
        return df
    
    def generate_signals(self, df):
        """Generate trading signals"""
        signals = []
        
        for i in range(3, len(df)):
            date = df.index[i]
            current = df.iloc[i]
            signal = None
            
            # Buy conditions
            if (current['rsi'] < self.rsi_buy_threshold and
                current['high_low_range'] > self.volatility_threshold and
                current['volume'] > current['volume_ma'] * self.volume_multiplier and
                current['low_3d_change'] < -self.trend_threshold):
                signal = 'buy'
            
            # Sell conditions
            elif (current['rsi'] > self.rsi_sell_threshold and
                  current['high_3d_change'] > self.trend_threshold and
                  current['low_3d_change'] > self.trend_threshold):
                signal = 'sell'
            
            if signal:
                signals.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'signal': signal,
                    'indicators': {
                        'rsi': round(current['rsi'], 2),
                        'volatility': round(current['high_low_range'], 4),
                        'volume_ratio': round(current['volume'] / current['volume_ma'], 2),
                        'trend': round(current['low_3d_change'], 4)
                    }
                })
        
        return signals
    
    def compare_with_original(self, generated_signals, original_file):
        """Compare generated signals with original signals"""
        # Read original signals
        with open(original_file, 'r') as f:
            content = f.read()
            import re
            pattern = r"{[\s]*date:[\s]*'([^']+)'[\s]*,[\s]*signal:[\s]*'([^']+)'[\s]*}"
            original_signals = re.findall(pattern, content)
        
        original_dict = {date: signal for date, signal in original_signals}
        generated_dict = {s['date']: s['signal'] for s in generated_signals}
        
        # Find matches and differences
        matches = 0
        for date in original_dict.keys():
            if date in generated_dict and original_dict[date] == generated_dict[date]:
                matches += 1
        
        total_original = len(original_dict)
        total_generated = len(generated_dict)
        
        return {
            'matches': matches,
            'total_original': total_original,
            'total_generated': total_generated,
            'accuracy': matches / total_original if total_original > 0 else 0
        }
    
    def run_test(self, price_file, original_signals_file=None):
        """Run complete test"""
        # Prepare data
        df = self.prepare_data(price_file)
        df = self.calculate_indicators(df)
        
        # Generate signals
        signals = self.generate_signals(df)
        
        # Save signals
        output_file = price_file.replace('.json', '_test_signals.js')
        with open(output_file, 'w') as f:
            f.write('[\n')
            for i, signal in enumerate(signals):
                f.write(f"  {{ date: '{signal['date']}', signal: '{signal['signal']}' }}")
                if i < len(signals) - 1:
                    f.write(',\n')
                else:
                    f.write('\n')
            f.write(']\n')
        
        # Save detailed analysis
        analysis_file = price_file.replace('.json', '_analysis.txt')
        with open(analysis_file, 'w') as f:
            f.write("Signal Analysis\n\n")
            for signal in signals:
                f.write(f"Date: {signal['date']}\n")
                f.write(f"Signal: {signal['signal'].upper()}\n")
                f.write("Indicators:\n")
                for name, value in signal['indicators'].items():
                    f.write(f"  {name}: {value}\n")
                f.write("\n")
        
        # Compare with original if provided
        if original_signals_file:
            comparison = self.compare_with_original(signals, original_signals_file)
            with open(analysis_file, 'a') as f:
                f.write("\nComparison with Original Signals:\n")
                f.write(f"Matched Records: {comparison['matches']}\n")
                f.write(f"Total Original: {comparison['total_original']}\n")
                f.write(f"Total Generated: {comparison['total_generated']}\n")
                f.write(f"Accuracy: {comparison['accuracy']*100:.2f}%\n")
            
            return comparison
        
        return len(signals)

if __name__ == "__main__":
    # Create tester with default parameters
    tester = TTMSignalTester()
    
    # Test with original data
    results = tester.run_test('originalCall.json', 'JustOutPut.js')
    
    print("\nTest Results:")
    print(f"Matched Records: {results['matches']}")
    print(f"Total Original: {results['total_original']}")
    print(f"Total Generated: {results['total_generated']}")
    print(f"Accuracy: {results['accuracy']*100:.2f}%")