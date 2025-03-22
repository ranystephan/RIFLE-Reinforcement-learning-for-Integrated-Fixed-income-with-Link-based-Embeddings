"""
Data Pipeline Script for Fixed Income RL Project

This script runs the complete data pipeline:
1. Fetching raw data
2. Processing and feature engineering
3. Preparing data for regime detection
4. Preparing data for RL agent
5. Saving processed data

Author: ranycs & cosrv
"""

import os
import argparse
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import sys

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.data_loader import FixedIncomeDataLoader
from src.data.data_processor import FixedIncomeDataProcessor
from src.data.data_utils import save_dataframe, save_pickle, ensure_dir

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the fixed income data pipeline')
    
    parser.add_argument('--fred-api-key', type=str, default=None,
                       help='FRED API key')
    parser.add_argument('--quandl-api-key', type=str, default=None,
                       help='Quandl API key')
    parser.add_argument('--start-date', type=str, default='2010-01-01',
                       help='Start date for data collection (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date for data collection (YYYY-MM-DD), defaults to yesterday')
    parser.add_argument('--n-lags', type=int, default=10,
                       help='Number of lags for time series features')
    parser.add_argument('--save-dir', type=str, default='data/processed',
                       help='Directory to save processed data')
    
    return parser.parse_args()

def main():
    """Run the data pipeline."""
    # Parse arguments
    args = parse_args()
    
    # Set default end date to yesterday if not provided
    if args.end_date is None:
        args.end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    logger.info(f"Running data pipeline from {args.start_date} to {args.end_date}")
    
    # Create data loader
    data_loader = FixedIncomeDataLoader(
        fred_api_key=args.fred_api_key,
        quandl_api_key=args.quandl_api_key
    )
    
    # Create data processor
    data_processor = FixedIncomeDataProcessor()
    
    # Step 1: Load raw data
    logger.info("Loading raw data...")
    raw_data = data_loader.load_all_data(
        start_date=args.start_date,
        end_date=args.end_date,
        save=True
    )
    
    # Step 2: Process data
    logger.info("Processing data...")
    processed_data = data_processor.process_data_pipeline(
        data_dict=raw_data,
        for_regime=True,
        for_rl=True,
        n_lags=args.n_lags
    )
    
    # Step 3: Save processed data
    logger.info("Saving processed data...")
    ensure_dir(args.save_dir)
    
    # Save regime detection features
    if 'regime_features' in processed_data:
        save_dataframe(
            processed_data['regime_features'],
            'regime_features.csv',
            args.save_dir
        )
        save_dataframe(
            processed_data['regime_features_normalized'],
            'regime_features_normalized.csv',
            args.save_dir
        )
        
        # Save train/val/test splits
        save_dataframe(
            processed_data['regime_train'],
            'regime_train.csv',
            args.save_dir
        )
        save_dataframe(
            processed_data['regime_val'],
            'regime_val.csv',
            args.save_dir
        )
        save_dataframe(
            processed_data['regime_test'],
            'regime_test.csv',
            args.save_dir
        )
    
    # Save RL features
    if 'rl_features' in processed_data:
        save_dataframe(
            processed_data['rl_features'],
            'rl_features.csv',
            args.save_dir
        )
        save_dataframe(
            processed_data['rl_features_normalized'],
            'rl_features_normalized.csv',
            args.save_dir
        )
        
        # Save train/val/test splits
        save_dataframe(
            processed_data['rl_train'],
            'rl_train.csv',
            args.save_dir
        )
        save_dataframe(
            processed_data['rl_val'],
            'rl_val.csv',
            args.save_dir
        )
        save_dataframe(
            processed_data['rl_test'],
            'rl_test.csv',
            args.save_dir
        )
    
    # Save scalers for later use
    save_pickle(
        data_processor.scalers,
        'feature_scalers.pkl',
        args.save_dir
    )
    
    logger.info("Data pipeline completed successfully.")

if __name__ == '__main__':
    main()
