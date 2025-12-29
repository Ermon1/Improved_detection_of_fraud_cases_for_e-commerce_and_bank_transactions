import pandas as pd
import numpy as np
import os
import argparse
import sys
from sklearn.preprocessing import StandardScaler
import joblib

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

def create_time_features(df):
    logger.info("Creating time-based features...")
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    
    # Time since signup in seconds
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
    
    # Hour of day
    df['hour_of_day'] = df['purchase_time'].dt.hour
    
    # Day of week
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    
    return df

def create_freq_features(df):
    logger.info("Creating frequency features...")
    
    # Device Frequency
    device_count = df.groupby('device_id')['user_id'].count().reset_index()
    device_count.columns = ['device_id', 'device_freq']
    df = pd.merge(df, device_count, on='device_id', how='left')
    
    # IP Frequency (using the original IP, not the range/country)
    # Note: If IP mapping was done, we still have ip_address column
    if 'ip_address' in df.columns:
        ip_count = df.groupby('ip_address')['user_id'].count().reset_index()
        ip_count.columns = ['ip_address', 'ip_freq']
        df = pd.merge(df, ip_count, on='ip_address', how='left')
    
    return df

def encode_features(df):
    logger.info("Encoding categorical features...")
    
    # Select categorical columns
    cat_cols = ['source', 'browser', 'sex', 'country']
    
    # One-Hot Encoding
    # We use drop_first=True to avoid dummy variable trap
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # We drop non-feature columns
    cols_to_drop = ['user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address']
    # Check if they exist before dropping
    cols_to_drop = [c for c in cols_to_drop if c in df_encoded.columns]
    
    df_encoded = df_encoded.drop(columns=cols_to_drop)
    
    return df_encoded

def scale_features(df):
    logger.info("Scaling numerical features...")
    
    # Identify numerical columns to scale
    # We shouldn't scale the target 'class' or the one-hot encoded binary columns
    # We only scale continuous features
    
    scale_cols = ['purchase_value', 'age', 'time_since_signup', 'device_freq', 'ip_freq']
    
    # Check availability
    scale_cols = [c for c in scale_cols if c in df.columns]
    
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    
    return df, scaler

def main():
    parser = argparse.ArgumentParser(description="Feature Engineering for Fraud Data")
    parser.add_argument('--base_dir', type=str, default=os.getcwd(), help='Base directory')
    args = parser.parse_args()
    
    base_dir = args.base_dir
    data_processed_dir = os.path.join(base_dir, 'data/processed')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    input_file = os.path.join(data_processed_dir, 'fraud_data_with_country.csv')
    
    try:
        df = load_data(input_file)
        
        df = create_time_features(df)
        df = create_freq_features(df)
        df_encoded = encode_features(df)
        df_scaled, scaler = scale_features(df_encoded)
        
        output_file = os.path.join(data_processed_dir, 'fraud_data_encoded.csv')
        df_scaled.to_csv(output_file, index=False)
        
        scaler_file = os.path.join(models_dir, 'scaler.pkl')
        joblib.dump(scaler, scaler_file)
        
        logger.info(f"Feature engineering complete. Saved to {output_file}")
        logger.info(f"Final shape: {df_scaled.shape}")
        
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
