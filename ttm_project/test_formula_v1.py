import json
import pandas as pd
from datetime import datetime

def generate_signals(json_file):
    """Apply the discovered formula to new price data"""
    # Read JSON file
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
    
    # Sort by date
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Calculate required indicators
    df['price_change'] = (df['close'] - df['open']) / df['open']
    
    # Calculate high and low changes
    df['high_change_1d'] = df['high'].pct_change()
    df['low_change_1d'] = df['low'].pct_change()
    df['high_change_3d'] = df['high'].pct_change(periods=3)
    df['low_change_3d'] = df['low'].pct_change(periods=3)
    
    # Generate signals
    signals = []
    for idx, row in df.iterrows():
        if pd.isna(row['high_change_3d']):  # Skip if we don't have enough history
            continue
            
        # Check buy conditions
        if (row['price_change'] > -0.0092 and
            row['low_change_3d'] > -0.0319 and
            row['high_change_3d'] > -0.0242 and
            row['high_change_1d'] > -0.0142 and
            row['low_change_1d'] > -0.0136):
            signals.append({
                'date': row['date'].strftime('%Y-%m-%d'),
                'signal': 'buy'
            })
            
        # Check sell conditions
        elif (row['price_change'] < 0.0052 and
              row['low_change_3d'] < 0.0304 and
              row['high_change_3d'] < 0.0258 and
              row['high_change_1d'] < 0.0109 and
              row['low_change_1d'] < 0.0147):
            signals.append({
                'date': row['date'].strftime('%Y-%m-%d'),
                'signal': 'sell'
            })
    
    # Save signals to file
    output_file = json_file.replace('.json', '_signals.js')
    with open(output_file, 'w') as f:
        f.write('[\n')
        for i, signal in enumerate(signals):
            f.write(f"  {{ date: '{signal['date']}', signal: '{signal['signal']}' }}")
            if i < len(signals) - 1:
                f.write(',\n')
            else:
                f.write('\n')
        f.write(']\n')
    
    print(f"Generated {len(signals)} signals")
    print(f"Results saved to {output_file}")
    
    # Also create a detailed CSV for analysis
    df['signal'] = 'none'
    for signal in signals:
        df.loc[df['date'].dt.strftime('%Y-%m-%d') == signal['date'], 'signal'] = signal['signal']
    
    analysis_file = json_file.replace('.json', '_analysis.csv')
    df.to_csv(analysis_file, index=False)
    print(f"Detailed analysis saved to {analysis_file}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_formula.py new_data.json")
        sys.exit(1)
    
    json_file = sys.argv[1]
    generate_signals(json_file)