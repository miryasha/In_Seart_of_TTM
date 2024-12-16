# evaluate_predictions.py
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def get_prediction_and_actual(start_date, end_date):
    """Get predictions and actual values for a date range"""
    print("Retrieving actual data from database...")
    engine = create_engine('mysql+pymysql://root:rootroot@localhost/money')
    
    query = text("""
        WITH numbered_rows AS (
            SELECT 
                starting_date,
                C_W,
                LEAD(C_W) OVER (ORDER BY starting_date) as next_actual_C_W
            FROM no_price_anaysis_tbl
            WHERE starting_date BETWEEN :start_date AND :end_date
        )
        SELECT *
        FROM numbered_rows
        ORDER BY starting_date;
    """)
    
    try:
        df = pd.read_sql(query, engine, params={"start_date": start_date, "end_date": end_date})
        print(f"Retrieved {len(df)} rows of actual data")
        return df
    finally:
        engine.dispose()

def calculate_confidence_metrics(evaluation_df):
    """Calculate additional confidence-based metrics"""
    confidence_thresholds = [0.6, 0.7, 0.8]
    
    print("\nConfidence Level Analysis:")
    for threshold in confidence_thresholds:
        high_conf_predictions = evaluation_df[
            ((evaluation_df['predicted_next_C_W'] == -1) & (evaluation_df['prob_down'] > threshold)) |
            ((evaluation_df['predicted_next_C_W'] == 1) & (evaluation_df['prob_up'] > threshold))
        ]
        
        if len(high_conf_predictions) > 0:
            high_conf_correct = (high_conf_predictions['predicted_next_C_W'] == 
                               high_conf_predictions['next_actual_C_W']).sum()
            high_conf_accuracy = high_conf_correct / len(high_conf_predictions)
            
            print(f"\nPredictions with >{threshold:.0%} confidence:")
            print(f"Total predictions: {len(high_conf_predictions)}")
            print(f"Correct predictions: {high_conf_correct}")
            print(f"Accuracy: {high_conf_accuracy:.2%}")

def evaluate_predictions(start_date, end_date):
    print(f"Evaluating predictions from {start_date} to {end_date}")
    
    # Get actual data
    df_actual = get_prediction_and_actual(start_date, end_date)
    if df_actual.empty:
        print("No actual data available for evaluation")
        return
        
    # Load prediction results
    try:
        predictions_df = pd.read_csv(f'predictions_{start_date}_to_{end_date}.csv')
        print(f"Loaded predictions file with {len(predictions_df)} rows")
    except FileNotFoundError:
        print(f"No prediction file found: predictions_{start_date}_to_{end_date}.csv")
        return
    
    # Convert dates to datetime for proper merging
    df_actual['starting_date'] = pd.to_datetime(df_actual['starting_date'])
    predictions_df['date'] = pd.to_datetime(predictions_df['date'])
    
    # Merge predictions with actual results
    evaluation_df = predictions_df.merge(
        df_actual[['starting_date', 'next_actual_C_W']], 
        left_on='date', 
        right_on='starting_date',
        how='inner'
    )
    
    # Remove rows where we don't have actual next week's values
    evaluation_df = evaluation_df.dropna(subset=['next_actual_C_W'])
    
    print(f"\nAnalyzing {len(evaluation_df)} predictions with known outcomes")
    
    # Calculate accuracy metrics
    correct_predictions = (evaluation_df['predicted_next_C_W'] == evaluation_df['next_actual_C_W']).sum()
    total_predictions = len(evaluation_df)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    # Print evaluation results
    print("\nEvaluation Results:")
    print(f"Total predictions evaluated: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2%}")
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(evaluation_df['next_actual_C_W'], 
                              evaluation_df['predicted_next_C_W']))
    
    # Print confusion matrix
    conf_matrix = confusion_matrix(evaluation_df['next_actual_C_W'], 
                                 evaluation_df['predicted_next_C_W'])
    print("\nConfusion Matrix:")
    print("Predicted  -1    1")
    print(f"Actual -1   {conf_matrix[0,0]}   {conf_matrix[0,1]}")
    print(f"Actual  1   {conf_matrix[1,0]}   {conf_matrix[1,1]}")
    
    # Detailed analysis by week
    print("\nDetailed Weekly Analysis:")
    for _, row in evaluation_df.iterrows():
        date = row['date'].strftime('%Y-%m-%d')
        prediction = int(row['predicted_next_C_W'])
        actual = int(row['next_actual_C_W'])
        prob_down = row['prob_down']
        prob_up = row['prob_up']
        correct = "✓" if prediction == actual else "✗"
        confidence = prob_down if prediction == -1 else prob_up
        print(f"Week of {date}:")
        print(f"  Predicted: {prediction:2d} | Actual: {actual:2d} | Confidence: {confidence:.2%} | {correct}")
    
    # Calculate additional metrics
    calculate_confidence_metrics(evaluation_df)
    
    # Calculate winning/losing streaks
    predictions_correct = evaluation_df['predicted_next_C_W'] == evaluation_df['next_actual_C_W']
    current_streak = 0
    max_streak = 0
    for is_correct in predictions_correct:
        if is_correct:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
            
    print(f"\nLongest streak of correct predictions: {max_streak}")
    
    # Save detailed results
    result_filename = f'evaluation_results_{start_date}_to_{end_date}.csv'
    evaluation_df.to_csv(result_filename, index=False)
    print(f"\nDetailed results saved to {result_filename}")
    
    # Print summary analysis
    print("\nKey Findings:")
    print(f"1. Overall Accuracy: {accuracy:.2%}")
    print(f"2. Up Movement Precision: {conf_matrix[1,1]/(conf_matrix[1,1] + conf_matrix[0,1]):.2%}")
    print(f"3. Down Movement Precision: {conf_matrix[0,0]/(conf_matrix[0,0] + conf_matrix[1,0]):.2%}")
    print(f"4. Predictions favor up movements: {conf_matrix[:,1].sum()}/{total_predictions} predictions")
    
    # Calculate prediction bias
    actual_up_ratio = (evaluation_df['next_actual_C_W'] == 1).mean()
    predicted_up_ratio = (evaluation_df['predicted_next_C_W'] == 1).mean()
    print(f"\nModel Bias Analysis:")
    print(f"Actual Up/Down Ratio: {actual_up_ratio:.2%} up vs {1-actual_up_ratio:.2%} down")
    print(f"Predicted Up/Down Ratio: {predicted_up_ratio:.2%} up vs {1-predicted_up_ratio:.2%} down")

if __name__ == "__main__":
    print("Model Evaluation Tool")
    print("Enter date range for evaluation:")
    start_date = input("Start date (YYYY-MM-DD): ")
    end_date = input("End date (YYYY-MM-DD): ")
    
    try:
        evaluate_predictions(start_date, end_date)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("\nDebug information:")
        print(f"Start date: {start_date}")
        print(f"End date: {end_date}")