import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import os
import argparse
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_resources(models_dir):
    logger.info(f"Loading resources from {models_dir}...")
    model_path = os.path.join(models_dir, 'best_fraud_model.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
        
    model_pipeline = joblib.load(model_path)
    return model_pipeline

def explain_model(model_pipeline, df, output_dir):
    logger.info("Generating SHAP explanations...")
    
    # Extract the actual model (last step of pipeline)
    # Pipeline steps: [('smote', ...), ('model', ...)]
    model = model_pipeline.named_steps['model']
    
    # Check if we can use TreeExplainer (faster)
    # If using sklearn models needing ImbPipeline, the input to model prediction
    # is the output of the previous steps.
    
    # However, SHAP TreeExplainer often needs the raw model and the data *as seen by the model*.
    # Since SMOTE is only used for training, for inference/explainability on *test* data,
    # we simulate the pipeline transformation (excluding resampling).
    
    # Actually, the pipeline for prediction calls 'predict' which usually skips samplers.
    # But for SHAP we need the input features X.
    
    # Let's take a sample of data for SHAP (it's slow on large datasets)
    sample_size = 1000
    if len(df) > sample_size:
        df_sample = df.sample(sample_size, random_state=42)
    else:
        df_sample = df
    
    logger.info(f"Using sample of size {len(df_sample)} for explainability.")

    # There are no transformations in the pipeline other than SMOTE (which is train-only)
    # So X_sample is directly passed to the model.
    X_sample = df_sample.drop(columns=['class']) if 'class' in df_sample.columns else df_sample
    
    try:
        explainer = shap.Explainer(model, X_sample)
    except Exception:
        # Fallback for models not supported by generic Explainer w/ mask
        # e.g. using KernelExplainer explicitly if needed, but Explainer(model, X) covers most.
        logger.warning("Generic Explainer failed, trying LinearExplainer explicitly or KernelExplainer.")
        if hasattr(model, 'coef_'):
            explainer = shap.LinearExplainer(model, X_sample)
        else:
             explainer = shap.KernelExplainer(model.predict_proba, X_sample)
    shap_values = explainer(X_sample)
    
    # 1. Summary Plot
    try:
        logger.info("Creating Summary Plot...")
        plt.figure()
        # Handle SHAP Explanation object vs raw matrix
        values = shap_values.values if hasattr(shap_values, 'values') else shap_values
        shap.summary_plot(values, X_sample, show=False)
        plt.savefig(os.path.join(output_dir, 'shap_summary_plot.png'), bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.error(f"Failed to create summary plot: {e}")
    
    # 2. Bar Plot (Global Importance)
    try:
        logger.info("Creating Bar Plot...")
        plt.figure()
        # For bar plot, if it's an Explanation object, it works directly. 
        # If it's array, we might need feature names.
        shap.plots.bar(shap_values, show=False)
        plt.savefig(os.path.join(output_dir, 'shap_bar_plot.png'), bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.error(f"Failed to create bar plot: {e}")
    
    logger.info(f"Plots saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate SHAP Explanations")
    parser.add_argument('--base_dir', type=str, default=os.getcwd(), help='Base directory')
    args = parser.parse_args()
    
    base_dir = args.base_dir
    models_dir = os.path.join(base_dir, 'models')
    data_processed_dir = os.path.join(base_dir, 'data/processed')
    reports_dir = os.path.join(base_dir, 'reports/figures')
    os.makedirs(reports_dir, exist_ok=True)
    
    input_file = os.path.join(data_processed_dir, 'fraud_data_encoded.csv')
    
    try:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Data not found at {input_file}")
            
        df = pd.read_csv(input_file)
        
        # Load Model
        model_pipeline = load_resources(models_dir)
        
        # Explain
        explain_model(model_pipeline, df, reports_dir)
        
    except Exception as e:
        logger.error(f"Error during explainability: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
