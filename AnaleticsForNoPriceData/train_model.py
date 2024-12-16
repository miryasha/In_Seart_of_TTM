# train_model.py
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
import joblib

def get_training_data():
    print("Connecting to database...")
    engine = create_engine('mysql+pymysql://root:rootroot@localhost/money')
    
    query = """
    WITH numbered_rows AS (
        SELECT 
            *,
            LEAD(C_W) OVER (ORDER BY starting_date) as next_C_W,
            LAG(C_W, 1) OVER (ORDER BY starting_date) as prev_C_W,
            LAG(C_W, 2) OVER (ORDER BY starting_date) as prev_2_C_W
        FROM no_price_anaysis_tbl
        WHERE starting_date <= '2024-01-01'
    )
    SELECT * FROM numbered_rows
    WHERE next_C_W IS NOT NULL;
    """
    
    try:
        df = pd.read_sql(query, engine)
        print(f"Retrieved {len(df)} rows for training")
        return df
    except Exception as e:
        print(f"Error retrieving data: {e}")
        return None
    finally:
        engine.dispose()

def create_advanced_features(df):
    """Create additional features for better prediction"""
    
    # Fill NaN values in lagged features
    df['prev_C_W'] = df['prev_C_W'].fillna(0)
    df['prev_2_C_W'] = df['prev_2_C_W'].fillna(0)
    
    # Create trend features
    df['weekly_trend'] = np.where(df['C_sum_ground'] > 0, 1, 
                                 np.where(df['C_sum_ground'] < 0, -1, 0))
    
    # Create momentum features
    df['momentum'] = df['C_W'] * abs(df['C_sum_ground'])
    
    # Create volatility feature
    df['day_changes'] = df[['C_1', 'C_2', 'C_3', 'C_4', 'C_5']].nunique(axis=1)
    
    # Create pattern features
    df['reversal_pattern'] = np.where(
        (df['C_sum_ground'].abs() >= 3) & (df['C_W'] != df['prev_C_W']), 1, 0
    )
    
    # Create consistency features
    df['direction_consistency'] = df[['C_1', 'C_2', 'C_3', 'C_4', 'C_5']].apply(
        lambda x: x.value_counts().max() / 5, axis=1
    )
    
    # Ensure no NaN values remain
    df = df.fillna(0)
    
    return df

def train_and_save_model():
    # Get data
    df = get_training_data()
    if df is None:
        return
    
    # Create advanced features
    print("Creating advanced features...")
    df = create_advanced_features(df)
    
    # Prepare features
    feature_columns = [
        'C_1', 'C_2', 'C_3', 'C_4', 'C_5', 'C_W', 
        'C_sum_positive', 'C_sum_negetive', 'C_sum_ground', 'C_multiply',
        'weekly_trend', 'momentum', 'day_changes', 'reversal_pattern',
        'direction_consistency', 'prev_C_W', 'prev_2_C_W'
    ]
    
    X = df[feature_columns]
    y = df['next_C_W']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train Random Forest
    print("\nTraining Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    rf_model.fit(X_train_scaled, y_train)
    
    # Evaluate Random Forest
    rf_pred = rf_model.predict(X_test_scaled)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    print("\nRANDOM FOREST Results:")
    print(f"Test accuracy: {rf_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, rf_pred))
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Save model and components
    print("\nSaving model and components...")
    joblib.dump(rf_model, 'price_prediction_model.joblib')
    joblib.dump(scaler, 'price_prediction_scaler.joblib')
    
    # Save feature configuration
    with open('model_config.txt', 'w') as f:
        f.write("Feature Columns:\n")
        f.write('\n'.join(feature_columns))
        f.write("\n\nModel Performance:\n")
        f.write(f"Test Accuracy: {rf_accuracy:.4f}\n")
        f.write("\nFeature Importance:\n")
        f.write(feature_importance.to_string())
    
    return rf_model, scaler, feature_columns

if __name__ == "__main__":
    print("Starting enhanced model training...")
    model, scaler, features = train_and_save_model()
    print("\nTraining complete!")