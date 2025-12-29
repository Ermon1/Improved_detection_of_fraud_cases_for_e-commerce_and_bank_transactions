import pandas as pd
import numpy as np
import os
import argparse
import sys
import json
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, average_precision_score, precision_recall_curve, auc, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(input_path):
    logger.info(f"Loading data from {input_path}...")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Data not found at {input_path}")
    df = pd.read_csv(input_path)
    return df

def train_and_evaluate(df):
    # Separate features and target
    target_col = 'class'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 1. Split data (Stratified)
    logger.info("Splitting data into Train and Test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    logger.info(f"Class distribution in Train: {y_train.value_counts(normalize=True).to_dict()}")
    
    # 2. Define Models (Simplified for faster execution)
    models = {
        'Logistic Regression': LogisticRegression(max_iter=100, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=10, max_depth=3, random_state=42, n_jobs=-1)
    }
    
    results = {}
    
    # K-Fold CV
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    
    for name, model in models.items():
        logger.info(f"Evaluating {name}...")
        
        # Pipeline with SMOTE
        # SMOTE is applied only on the training folds during CV
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('model', model)
        ])
        
        # Metrics to evaluate
        scoring = {'f1': 'f1', 'auc_pr': 'average_precision', 'roc_auc': 'roc_auc'}
        
        cv_results = cross_validate(pipeline, X_train, y_train, cv=skf, scoring=scoring, n_jobs=-1)
        
        mean_f1 = np.mean(cv_results['test_f1'])
        std_f1 = np.std(cv_results['test_f1'])
        mean_auc_pr = np.mean(cv_results['test_auc_pr'])
        
        results[name] = {
            'cv_mean_f1': mean_f1,
            'cv_std_f1': std_f1,
            'cv_mean_auc_pr': mean_auc_pr
        }
        
        logger.info(f"{name} - CV F1: {mean_f1:.4f} (+/- {std_f1:.4f}), AUC-PR: {mean_auc_pr:.4f}")

    # 3. Select Best Model
    best_model_name = max(results, key=lambda k: results[k]['cv_mean_f1'])
    logger.info(f"Best model based on F1: {best_model_name}")
    
    # 4. Train Best Model on full Train set and Evaluate on Test
    logger.info(f"Training best model ({best_model_name}) on full training set...")
    
    best_model = models[best_model_name]
    final_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('model', best_model)
    ])
    
    final_pipeline.fit(X_train, y_train)
    
    logger.info("Evaluating on Test set...")
    y_pred = final_pipeline.predict(X_test)
    y_prob = final_pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate Test Metrics
    test_f1 = f1_score(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    test_auc_pr = auc(recall, precision)
    cm = confusion_matrix(y_test, y_pred)
    
    results['test_metrics'] = {
        'best_model': best_model_name,
        'f1_score': test_f1,
        'auc_pr': test_auc_pr,
        'confusion_matrix': cm.tolist()
    }
    
    logger.info(f"Test F1: {test_f1:.4f}")
    logger.info(f"Test AUC-PR: {test_auc_pr:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    
    return final_pipeline, results

def main():
    parser = argparse.ArgumentParser(description="Train Fraud Detection Models")
    parser.add_argument('--base_dir', type=str, default=os.getcwd(), help='Base directory')
    args = parser.parse_args()
    
    base_dir = args.base_dir
    data_processed_dir = os.path.join(base_dir, 'data/processed')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    input_file = os.path.join(data_processed_dir, 'fraud_data_encoded.csv')
    
    try:
        df = load_data(input_file)
        
        best_model, results = train_and_evaluate(df)
        
        # Save Model
        model_path = os.path.join(models_dir, 'best_fraud_model.pkl')
        joblib.dump(best_model, model_path)
        logger.info(f"Saved best model to {model_path}")
        
        # Save Results
        results_path = os.path.join(models_dir, 'model_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Saved results to {results_path}")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        # traceback
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
