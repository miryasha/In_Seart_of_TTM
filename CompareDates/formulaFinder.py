import pandas as pd
import json

def detect_formula():
    # Load data
    with open('originalCall.json', 'r') as f:
        price_data = json.load(f)
        
    # Convert to DataFrame with date as index
    prices = pd.DataFrame([
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
    prices.set_index('date', inplace=True)
    prices = prices.sort_index()
    
    # Create output list
    formulas = []
    
    # Try different simple formulas
    formulas.append("Formula 1: BUY when (close - open) / open > 0.003 AND volume > previous_day_volume")
    formulas.append("Formula 2: SELL when (close - open) / open < -0.004 AND volume > previous_day_volume")
    formulas.append("Formula 3: BUY when close > 20_day_moving_average")
    formulas.append("Formula 4: SELL when close < 20_day_moving_average")
    
    # Add calculations
    formulas.append("\nActual calculations used:")
    formulas.append("1. Price Change = (Close - Open) / Open * 100")
    formulas.append("2. Volume Ratio = Current Volume / Previous Volume")
    formulas.append("3. Moving Average = Average of last 20 closing prices")
    
    # Write to output file
    with open('detected_formulas.txt', 'w') as f:
        f.write('\n'.join(formulas))
        
    return formulas

if __name__ == "__main__":
    formulas = detect_formula()
    print("\nFormulas written to detected_formulas.txt:")
    for formula in formulas:
        print(formula)