import json
import pandas as pd

def generate_ttm_signals(json_file, output_file, min_swing=0.0):
    # Read JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
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
        for date, values in data['pricesInfo']['priceDataObj'].items()
    ])
    
    # Sort by date
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Get previous closes
    df['close_1'] = df['close'].shift(1)  # Previous close
    df['close_2'] = df['close'].shift(2)  # Two bars ago close
    
    signals = []
    
    # Loop through data starting from the third bar
    for i in range(2, len(df)):
        current_row = df.iloc[i]
        prev_row = df.iloc[i-1]
        prev2_row = df.iloc[i-2]
        
        # Check for buy signal (3 higher closes)
        if (current_row['close'] > prev_row['close'] > prev2_row['close']):
            signals.append({
                'date': prev2_row['date'].strftime('%Y-%m-%d'),  # First bar in the series
                'signal': 'buy'
            })
        
        # Check for sell signal (3 lower closes)
        elif (current_row['close'] < prev_row['close'] < prev2_row['close']):
            signals.append({
                'date': prev2_row['date'].strftime('%Y-%m-%d'),  # First bar in the series
                'signal': 'sell'
            })
    
    # Write to output file
    with open(output_file, 'w') as f:
        f.write('[\n')
        for i, signal in enumerate(signals):
            f.write(f"  {{ date: '{signal['date']}', signal: '{signal['signal']}' }}")
            if i < len(signals) - 1:
                f.write(',\n')
            else:
                f.write('\n')
        f.write(']\n')
    
    print(f"TTM Scalper signals saved to {output_file}")

if __name__ == "__main__":
    input_file = "originalCall.json"
    output_file = "ttm_signals.js"
    generate_ttm_signals(input_file, output_file)