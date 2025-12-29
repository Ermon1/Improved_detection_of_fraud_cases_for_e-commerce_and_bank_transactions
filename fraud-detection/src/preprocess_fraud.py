import pandas as pd
import numpy as np
import os
import argparse
import sys
from datetime import datetime

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(fraud_path, ip_path):
    logger.info("Loading data...")
    if not os.path.exists(fraud_path):
        raise FileNotFoundError(f"Fraud data not found at {fraud_path}")
    if not os.path.exists(ip_path):
        raise FileNotFoundError(f"IP data not found at {ip_path}")
        
    df_fraud = pd.read_csv(fraud_path)
    df_ip = pd.read_csv(ip_path)
    logger.info(f"Loaded {len(df_fraud)} fraud records and {len(df_ip)} IP ranges.")
    return df_fraud, df_ip

def clean_data(df):
    logger.info("Cleaning fraud data...")
    # Convert timestamps
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    
    # Check for duplicates
    initial_len = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_len:
        logger.info(f"Dropped {initial_len - len(df)} duplicate rows.")
        
    return df

def efficient_ip_mapping(df_fraud, df_ip):
    logger.info("Starting efficient IP to Country mapping...")
    
    # Ensure IPs are integers
    # df_fraud['ip_address'] might be float
    fraud_ips = df_fraud['ip_address'].astype(int).values
    
    # Prepare IP ranges
    # Sort just in case
    df_ip = df_ip.sort_values('lower_bound_ip_address').reset_index(drop=True)
    
    lower_bounds = df_ip['lower_bound_ip_address'].astype(int).values
    upper_bounds = df_ip['upper_bound_ip_address'].astype(int).values
    countries = df_ip['country'].values
    
    # We can use binary search (searchsorted) to find the insertion point
    # searchsorted finds indices where elements should be inserted to maintain order
    # side='right' returns the index i such that a[i-1] <= v < a[i]
    
    logger.info("Performing binary search...")
    # This gives us the index of the range that starts *after* our IP
    # So we want the index - 1
    indices = np.searchsorted(lower_bounds, fraud_ips, side='right') - 1
    
    # Now verify if the IP is actually within the upper bound of that range
    # indices can be -1 if IP is smaller than the smallest lower bound
    
    assigned_countries = []
    
    # Vectorized verification
    # Create an array of 'Unknown'
    result_countries = np.full(len(fraud_ips), 'Unknown', dtype=object)
    
    # Valid indices filter: index >= 0 and index < len(df_ip)
    valid_mask = (indices >= 0) & (indices < len(df_ip))
    
    # Check upper bounds for potentially valid matches
    # We only check where valid_mask is True
    valid_indices = indices[valid_mask]
    current_ips = fraud_ips[valid_mask]
    
    # Check if IP <= upper_bound for the found range
    matched_upper = upper_bounds[valid_indices]
    in_range_mask = current_ips <= matched_upper
    
    # Where in_range_mask is True, we have a match
    final_indices = valid_indices[in_range_mask]
    
    # Update the result array
    # We need the original indices where valid_mask AND in_range_mask are True
    # Let's reconstruct the mask for the original array
    
    # Create a full boolean mask for the original array
    # consistent with valid_mask, but refined by in_range_mask
    final_mask = np.zeros(len(fraud_ips), dtype=bool)
    
    # We need to map back to original positions. 
    # valid_mask indices are where we looked.
    # in_range_mask are the subset of valid_mask that matched.
    
    # Use numpy indexing magic
    # Get the indices of the original array where valid_mask is True
    valid_original_indices = np.where(valid_mask)[0]
    
    # Of those, keep only the ones where in_range_mask is True
    success_original_indices = valid_original_indices[in_range_mask]
    
    # Assign countries
    result_countries[success_original_indices] = countries[final_indices]
    
    df_fraud['country'] = result_countries
    
    logger.info(f"Mapping complete. Found countries for {(result_countries != 'Unknown').sum()} records.")
    return df_fraud

def main():
    parser = argparse.ArgumentParser(description="Preprocess Fraud Data")
    parser.add_argument('--base_dir', type=str, default=os.getcwd(), help='Base directory')
    args = parser.parse_args()
    
    base_dir = args.base_dir
    data_raw_dir = os.path.join(base_dir, 'data/raw')
    data_processed_dir = os.path.join(base_dir, 'data/processed')
    
    os.makedirs(data_processed_dir, exist_ok=True)
    
    fraud_file = os.path.join(data_raw_dir, 'Fraud_Data.csv')
    ip_file = os.path.join(data_raw_dir, 'IpAddress_to_Country.csv')
    
    try:
        df_fraud, df_ip = load_data(fraud_file, ip_file)
        
        df_fraud = clean_data(df_fraud)
        
        df_processed = efficient_ip_mapping(df_fraud, df_ip)
        
        output_file = os.path.join(data_processed_dir, 'fraud_data_with_country.csv')
        df_processed.to_csv(output_file, index=False)
        logger.info(f"Saved processed data to {output_file}")
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
