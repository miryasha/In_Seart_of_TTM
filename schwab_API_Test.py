import requests
import json
import pandas as pd
from datetime import datetime
import numpy as np

class SchwabAPI:
    def __init__(self, api_key, api_secret, account_id):
        self.base_url = "https://api.schwab.com/v1"  # Example URL
        self.api_key = api_key
        self.api_secret = api_secret
        self.account_id = account_id
        self.session = requests.Session()
        self.authenticate()
    
    def authenticate(self):
        """Authenticate with Schwab API"""
        auth_url = f"{self.base_url}/oauth/token"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {
            "grant_type": "client_credentials",
            "client_id": self.api_key,
            "client_secret": self.api_secret
        }
        
        response = self.session.post(auth_url, headers=headers, data=data)
        if response.status_code == 200:
            self.session.headers.update({
                "Authorization": f"Bearer {response.json()['access_token']}"
            })
        else:
            raise Exception("Authentication failed")

    def get_historical_data(self, symbol, timeframe='1D', limit=100):
        """Get historical price data"""
        endpoint = f"{self.base_url}/markets/history"
        params = {
            "symbol": symbol,
            "interval": timeframe,
            "limit": limit
        }
        response = self.session.get(endpoint, params=params)
        return pd.DataFrame(response.json()['candles'])

    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        # Calculate EMA
        df['EMA9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['EMA21'] = df['close'].ewm(span=21, adjust=False).mean()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df

    def generate_signals(self, df):
        """Generate trading signals based on indicators"""
        df['Signal'] = 0
        
        # Crossover signals
        df['Signal'] = np.where(
            (df['EMA9'] > df['EMA21']) & (df['RSI'] < 70), 1,
            np.where((df['EMA9'] < df['EMA21']) & (df['RSI'] > 30), -1, 0)
        )
        
        return df

    def place_order(self, symbol, side, quantity, order_type='MARKET'):
        """Place a trading order"""
        endpoint = f"{self.base_url}/accounts/{self.account_id}/orders"
        data = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "type": order_type
        }
        response = self.session.post(endpoint, json=data)
        return response.json()

    def get_account_info(self):
        """Get account information"""
        endpoint = f"{self.base_url}/accounts/{self.account_id}"
        response = self.session.get(endpoint)
        return response.json()

def main():
    # Initialize API with your credentials
    api = SchwabAPI(
        api_key="YOUR_API_KEY",
        api_secret="YOUR_API_SECRET",
        account_id="YOUR_ACCOUNT_ID"
    )
    
    # Example usage
    symbol = "AAPL"
    
    # Get historical data and calculate indicators
    df = api.get_historical_data(symbol)
    df = api.calculate_indicators(df)
    df = api.generate_signals(df)
    
    # Check for new signals
    latest = df.iloc[-1]
    if latest['Signal'] == 1:
        print(f"Buy Signal for {symbol}")
        # Place buy order
        order = api.place_order(symbol, "BUY", 100)
        print(f"Order placed: {order}")
    elif latest['Signal'] == -1:
        print(f"Sell Signal for {symbol}")
        # Place sell order
        order = api.place_order(symbol, "SELL", 100)
        print(f"Order placed: {order}")

if __name__ == "__main__":
    main()