import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import json
import re
from datetime import datetime

class FormulaDiscovery:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.importance_weights = None
        
    def prepare_data(self, price_file, signal_file):
        """Load and prepare data with extensive feature engineering"""
        # Load price data
        with open(price_file, 'r') as f:
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
        
        # Calculate basic features
        df['price_change'] = (df['close'] - df['open']) / df['open']
        df['high_low_range'] = (df['high'] - df['low']) / df['low']
        df['volume_change'] = df['volume'].pct_change()
        
        # Calculate multi-period features
        for period in [1, 2, 3]:
            # Price changes
            df[f'close_change_{period}d'] = df['close'].pct_change(period)
            df[f'high_change_{period}d'] = df['high'].pct_change(period)
            df[f'low_change_{period}d'] = df['low'].pct_change(period)
            
            # Volume changes
            df[f'volume_ma_{period}'] = df['volume'].rolling(period).mean()
            
            # Price ranges
            df[f'high_low_range_{period}d'] = df['high'].rolling(period).max() - df['low'].rolling(period).min()
            
        # Calculate moving averages
        for period in [5, 10, 20]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}']
        
        # Load and add signals
        with open(signal_file, 'r') as f:
            content = f.read()
            pattern = r"{[\s]*date:[\s]*'([^']+)'[\s]*,[\s]*signal:[\s]*'([^']+)'[\s]*}"
            signals = re.findall(pattern, content)
            
        signal_df = pd.DataFrame(signals, columns=['date', 'signal'])
        signal_df['date'] = pd.to_datetime(signal_df['date'])
        
        # Merge signals
        df['signal'] = 'none'
        for idx, row in signal_df.iterrows():
            df.loc[df['date'] == row['date'], 'signal'] = row['signal']
        
        # Convert signals to numeric
        df['signal_numeric'] = df['signal'].map({'none': 0, 'buy': 1, 'sell': 2})
        
        return df.dropna()
    
    def build_model(self, input_shape):
        """Build a model that can help identify important features"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def analyze_feature_importance(self, X, y, feature_names):
        """Analyze which features are most important for signal generation"""
        # Train the model multiple times to get stable importance scores
        importance_scores = np.zeros(len(feature_names))
        n_iterations = 5
        
        for _ in range(n_iterations):
            # Train model
            self.model.fit(X, y, epochs=50, batch_size=32, verbose=0)
            
            # Get feature importance through permutation
            base_accuracy = self.model.evaluate(X, y, verbose=0)[1]
            
            for i in range(len(feature_names)):
                # Permute feature
                X_permuted = X.copy()
                X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
                
                # Measure accuracy drop
                new_accuracy = self.model.evaluate(X_permuted, y, verbose=0)[1]
                importance_scores[i] += (base_accuracy - new_accuracy)
        
        # Average the importance scores
        importance_scores /= n_iterations
        
        # Create importance dictionary
        self.importance_weights = dict(zip(feature_names, importance_scores))
        
        return self.importance_weights
    
    def discover_formula(self, price_file, signal_file):
        """Main method to discover the formula"""
        print("Loading and preparing data...")
        df = self.prepare_data(price_file, signal_file)
        
        # Prepare features for model
        feature_columns = [col for col in df.columns if col not in ['date', 'signal', 'signal_numeric']]
        X = df[feature_columns].values
        y = df['signal_numeric'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        print("Building and training model...")
        self.model = self.build_model(X_scaled.shape[1])
        
        print("Analyzing feature importance...")
        importance_dict = self.analyze_feature_importance(X_scaled, y, feature_columns)
        
        # Sort features by importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        print("\nMost important features for signal generation:")
        for feature, importance in sorted_features[:5]:
            print(f"{feature}: {importance:.4f}")
        
        # Analyze patterns in top features
        print("\nAnalyzing patterns in top features...")
        top_features = [f[0] for f in sorted_features[:5]]
        
        patterns = self.analyze_patterns(df, top_features)
        return patterns
    
    def analyze_patterns(self, df, top_features):
        """Analyze specific patterns in the most important features"""
        patterns = {
            'buy': {},
            'sell': {}
        }
        
        for signal_type in ['buy', 'sell']:
            signal_df = df[df['signal'] == signal_type]
            
            for feature in top_features:
                values = signal_df[feature]
                patterns[signal_type][feature] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max()
                }
        
        # Generate formula suggestions
        buy_conditions = []
        sell_conditions = []
        
        for feature in top_features:
            buy_mean = patterns['buy'][feature]['mean']
            buy_std = patterns['buy'][feature]['std']
            sell_mean = patterns['sell'][feature]['mean']
            sell_std = patterns['sell'][feature]['std']
            
            buy_conditions.append(f"{feature} > {buy_mean-buy_std:.4f}")
            sell_conditions.append(f"{feature} < {sell_mean+sell_std:.4f}")
        
        formula = f"""
Suggested Formula:
BUY when ALL of:
    {' AND '.join(buy_conditions)}

SELL when ALL of:
    {' AND '.join(sell_conditions)}
"""
        return formula

if __name__ == "__main__":
    discoverer = FormulaDiscovery()
    formula = discoverer.discover_formula('originalCall.json', 'JustOutPut.js')
    
    print("\nDiscovered Formula:")
    print(formula)
    
    # Save formula to file
    with open('discovered_formula.txt', 'w') as f:
        f.write(formula)