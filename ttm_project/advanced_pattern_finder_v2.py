import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
import json
from ta.trend import (
    EMAIndicator, MACD, ADXIndicator, 
    CCIIndicator
)
from ta.momentum import (
    RSIIndicator, StochasticOscillator, 
    WilliamsRIndicator
)
from ta.volume import (
    VolumeWeightedAveragePrice, 
    OnBalanceVolumeIndicator,
    ChaikinMoneyFlowIndicator
)
from ta.volatility import (
    BollingerBands, 
    AverageTrueRange
)

class EnhancedPatternFinder:
    def __init__(self, sequence_length=10, pivot_lookback=2):
        self.sequence_length = sequence_length
        self.pivot_lookback = pivot_lookback
        self.scaler = StandardScaler()
        self.model = None
        self.feature_importance = None
    
    def calculate_pivot_points(self, df):
        """Calculate Pivot High and Pivot Low points with enhanced metrics"""
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
        
        # Add pivot strength metrics
        df['pivot_high_strength'] = df['pivot_high'].rolling(window=5).max() / df['high']
        df['pivot_low_strength'] = df['low'] / df['pivot_low'].rolling(window=5).min()
        
        numeric_columns = [
            'distance_to_pivot_high', 'distance_to_pivot_low', 
            'pivot_range', 'pivot_range_pct', 'pivot_position',
            'pivot_high_strength', 'pivot_low_strength'
        ]
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        return df
    
    def add_technical_indicators(self, df):
        """Add enhanced technical indicators"""
        # Price Changes
        df['price_change'] = df['close'].pct_change()
        df['high_low_range'] = (df['high'] - df['low']) / df['low']
        
        # Moving Averages
        for period in [5, 8, 13, 21, 34, 55]:
            df[f'ema_{period}'] = EMAIndicator(close=df['close'], window=period).ema_indicator()
        
        # MACD
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # RSI and Stochastic
        df['rsi'] = RSIIndicator(close=df['close']).rsi()
        stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        df['williams_r'] = WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close']).williams_r()
        
        # Volume Indicators
        df['volume_change'] = df['volume'].pct_change()
        df['vwap'] = VolumeWeightedAveragePrice(
            high=df['high'], low=df['low'], 
            close=df['close'], volume=df['volume']
        ).volume_weighted_average_price()
        df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
        df['cmf'] = ChaikinMoneyFlowIndicator(
            high=df['high'], low=df['low'], 
            close=df['close'], volume=df['volume']
        ).chaikin_money_flow()
        
        # Volatility Indicators
        bb = BollingerBands(close=df['close'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
        
        df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
        df['cci'] = CCIIndicator(high=df['high'], low=df['low'], close=df['close']).cci()
        
        # Pattern Detection
        for period in [1, 2, 3, 5]:
            df[f'high_{period}d_change'] = df['high'].pct_change(periods=period)
            df[f'low_{period}d_change'] = df['low'].pct_change(periods=period)
        
        # Trend strength and market regime
        df['trend_strength'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
        df['volatility'] = df['close'].pct_change().rolling(window=20).std()
        df['adx'] = ADXIndicator(high=df['high'], low=df['low'], close=df['close']).adx()
        
        return df
    
    def calculate_risk_metrics(self, df):
        """Calculate risk management metrics"""
        df['stop_loss'] = np.where(
            df['signal'] == 'buy',
            df['close'] - (df['atr'] * 2),
            df['close'] + (df['atr'] * 2)
        )
        
        # Calculate potential targets based on risk-reward ratios
        df['target_1r'] = np.where(
            df['signal'] == 'buy',
            df['close'] + (df['close'] - df['stop_loss']),
            df['close'] - (df['stop_loss'] - df['close'])
        )
        
        df['target_2r'] = np.where(
            df['signal'] == 'buy',
            df['close'] + (df['close'] - df['stop_loss']) * 2,
            df['close'] - (df['stop_loss'] - df['close']) * 2
        )
        
        return df
    
    def calculate_signal_confidence(self, row):
        """Calculate confidence score for signals"""
        confidence = 0
        
        if row['signal'] == 'buy':
            # Technical indicator confirmations for buy
            if row['rsi'] < 30: confidence += 0.2
            if row['macd'] > row['macd_signal']: confidence += 0.15
            if row['close'] > row['bb_mid']: confidence += 0.15
            if row['cmf'] < -0.1: confidence += 0.15
            if row['stoch_k'] < 20: confidence += 0.15
            if row['cci'] < -100: confidence += 0.1
            if row['williams_r'] < -80: confidence += 0.1
            
        elif row['signal'] == 'sell':
            # Technical indicator confirmations for sell
            if row['rsi'] > 70: confidence += 0.2
            if row['macd'] < row['macd_signal']: confidence += 0.15
            if row['close'] < row['bb_mid']: confidence += 0.15
            if row['cmf'] > 0.1: confidence += 0.15
            if row['stoch_k'] > 80: confidence += 0.15
            if row['cci'] > 100: confidence += 0.1
            if row['williams_r'] > -20: confidence += 0.1
        
        return confidence
    
    def prepare_data(self, json_file, signal_file):
        """Prepare and enhance data with all indicators"""
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
        
        # Add all technical indicators
        df = self.add_technical_indicators(df)
        df = self.calculate_pivot_points(df)
        
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
        
        # Add risk metrics
        df = self.calculate_risk_metrics(df)
        
        # Calculate signal confidence
        df['signal_confidence'] = df.apply(self.calculate_signal_confidence, axis=1)
        
        # Fill NaN values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        return df


    def build_model(self, input_shape, output_shape):
        """Build enhanced LSTM model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(output_shape, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
    
    def create_sequences(self, df):
        """Create sequences for LSTM with enhanced feature selection"""
        feature_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in feature_cols if col not in ['date']]
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[feature_cols])
        
        X, y = [], []
        for i in range(len(df) - self.sequence_length):
            sequence = scaled_features[i:(i + self.sequence_length)]
            label = df['signal'].iloc[i + self.sequence_length]
            X.append(sequence)
            y.append(label)
        
        X = np.array(X, dtype=np.float32)
        y = pd.get_dummies(y).values
        
        return X, y, feature_cols
    
    def train_with_cross_validation(self, X, y, n_splits=5):
        """Train model with time series cross-validation"""
        kfold = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            print(f"\nTraining fold {fold + 1}/{n_splits}")
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = self.build_model(
                input_shape=(X.shape[1], X.shape[2]),
                output_shape=y.shape[1]
            )
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=32,
                verbose=1
            )
            
            # Evaluate on validation set
            val_scores = model.evaluate(X_val, y_val, verbose=0)
            score_dict = dict(zip(model.metrics_names, val_scores))
            scores.append(score_dict)
            
            # Save the best model based on validation accuracy
            if fold == 0 or scores[-1].get('val_accuracy', 0) > max(s.get('val_accuracy', 0) for s in scores[:-1]):
                self.model = model
        
        return scores

    def analyze_feature_importance(self, X, y, feature_cols):
        """Analyze feature importance using Random Forest"""
        # Reshape X for Random Forest (from 3D to 2D)
        n_samples = X.shape[0]
        X_reshaped = X.reshape(n_samples, -1)
        
        # Convert one-hot encoded y back to labels
        y_labels = np.argmax(y, axis=1)
        
        # Train Random Forest
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_reshaped, y_labels)
        
        # Calculate feature importance for each time step
        feature_importance = []
        n_features = X.shape[2]
        n_timesteps = X.shape[1]
        
        for t in range(n_timesteps):
            for f in range(n_features):
                importance = rf_classifier.feature_importances_[t * n_features + f]
                feature_importance.append({
                    'timestep': t,
                    'feature': feature_cols[f],
                    'importance': importance
                })
        
        # Sort by importance
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        return feature_importance

    def analyze_patterns(self, df):
        """Analyze patterns in the data with enhanced metrics"""
        patterns = {signal_type: {} for signal_type in ['buy', 'sell']}
        
        for signal_type in patterns.keys():
            signal_data = df[df['signal'] == signal_type]
            
            if not signal_data.empty:
                # Analyze all numeric columns
                for col in df.select_dtypes(include=[np.number]).columns:
                    if col not in ['date', 'signal'] and not signal_data[col].isnull().all():
                        values = signal_data[col].dropna()
                        if not values.empty:
                            patterns[signal_type][col] = {
                                'mean': values.mean(),
                                'std': values.std(),
                                'min': values.min(),
                                'max': values.max(),
                                'median': values.median(),
                                'q1': values.quantile(0.25),
                                'q3': values.quantile(0.75)
                            }
                
                # Add specific pattern analysis
                patterns[signal_type]['pattern_analysis'] = {
                    'avg_confidence': signal_data['signal_confidence'].mean(),
                    'avg_risk_reward': abs(
                        (signal_data['target_2r'] - signal_data['close']) /
                        (signal_data['close'] - signal_data['stop_loss'])
                    ).mean(),
                    'avg_win_rate': (
                        signal_data[signal_data['signal_confidence'] > 0.6]
                        .groupby('signal')['signal'].count() /
                        len(signal_data)
                    ).mean()
                }
        
        return patterns

    def find_patterns(self, json_file, signal_file):
        """Main method to find patterns with enhanced analysis"""
        print("Loading and preparing data...")
        df = self.prepare_data(json_file, signal_file)
        
        print("\nAnalyzing technical patterns...")
        patterns = self.analyze_patterns(df)
        
        print("\nPreparing sequences for LSTM...")
        X, y, feature_cols = self.create_sequences(df)
        
        print("\nTraining model with cross-validation...")
        cv_scores = self.train_with_cross_validation(X, y)
        
        print("\nAnalyzing feature importance...")
        feature_importance = self.analyze_feature_importance(X, y, feature_cols)
        self.feature_importance = feature_importance
        
        # Save results
        self.save_findings(patterns, df, cv_scores, feature_importance)
        
        return patterns, cv_scores, feature_importance

    def save_findings(self, patterns, df, cv_scores, feature_importance):
        """Save the discovered patterns and analysis results"""
        with open('enhanced_patterns_analysis.txt', 'w') as f:
            f.write("Enhanced Pattern Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Write pattern analysis
            for signal_type in ['buy', 'sell']:
                f.write(f"\n{signal_type.upper()} Patterns:\n")
                f.write("-" * 30 + "\n")
                
                if patterns[signal_type]:
                    # Write technical indicator patterns
                    for indicator, stats in patterns[signal_type].items():
                        if isinstance(stats, dict) and 'mean' in stats:
                            f.write(f"\n{indicator}:\n")
                            f.write(f"  Mean: {stats['mean']:.4f}\n")
                            f.write(f"  Median: {stats['median']:.4f}\n")
                            f.write(f"  Range: {stats['min']:.4f} to {stats['max']:.4f}\n")
                            f.write(f"  IQR: {stats['q1']:.4f} to {stats['q3']:.4f}\n")
                            f.write(f"  Typical range: {stats['mean'] - stats['std']:.4f} to {stats['mean'] + stats['std']:.4f}\n")
                        
                        elif indicator == 'pattern_analysis':
                            f.write("\nPattern Analysis Metrics:\n")
                            for metric, value in stats.items():
                                f.write(f"  {metric}: {value:.4f}\n")
            
            # Write cross-validation results
            f.write("\nCross-Validation Results:\n")
            f.write("-" * 30 + "\n")
            avg_scores = {
                metric: np.mean([score[metric] for score in cv_scores])
                for metric in cv_scores[0].keys()
            }
            for metric, value in avg_scores.items():
                f.write(f"{metric}: {value:.4f}\n")
            
            # Write top feature importance
            f.write("\nTop Feature Importance:\n")
            f.write("-" * 30 + "\n")
            for i, feat in enumerate(feature_importance[:20]):
                f.write(f"{i+1}. {feat['feature']} (t-{feat['timestep']}): {feat['importance']:.4f}\n")

if __name__ == "__main__":
    finder = EnhancedPatternFinder()
    patterns, cv_scores, feature_importance = finder.find_patterns(
        'originalCall.json',
        'JustOutPut.js'
    )
    
    print("\nPattern analysis complete! Check enhanced_patterns_analysis.txt for results.")