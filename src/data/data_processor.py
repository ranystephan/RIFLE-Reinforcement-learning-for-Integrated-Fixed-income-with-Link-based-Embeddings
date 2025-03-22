"""
Data Processor Module for Fixed Income RL Project

This module handles preprocessing and feature engineering for fixed income data:
1. Aligning different datasets on the same date index
2. Computing derived features
3. Preparing data for the regime detection model
4. Creating features for the RL agent

Author: ranycs & cosrv

Technical explanation of this code:
1. Aligning different datasets on the same date index:
    - This is done to ensure that all datasets are aligned on the same date index.
    - This is important because it allows us to compare the datasets on the same date index.
    - This is also important because it allows us to use the same date index for the regime detection model and the RL agent.
    
2. Computing derived features:
    - This is done to create new features from the raw data.
    - These features are then used to train the regime detection model and the RL agent.
    - The derived features are computed in the compute_derived_features method.
    
3. Preparing data for the regime detection model:
    - This is done to prepare the data for the regime detection model.
    - The data is prepared in the prepare_regime_detection_features method.
    
4. Preparing data for the RL agent:
    - This is done to prepare the data for the RL agent.
    - The data is prepared in the prepare_rl_features method.
    
5. Normalizing features:
    - This is done to normalize the features.
    - The features are normalized in the normalize_features method.
    

Regime Detection Model:
    - This is a model that is used to detect the regime of the data.
    - The regime detection model is trained in the train_regime_detection_model method.
    - The regime detection model is then used to predict the regime of the data in the predict_regime method.
    

Regime means the state of the market. 
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from scipy import stats
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FixedIncomeDataProcessor:
    """
    Class for processing fixed income data for the RL project.
    """
    
    def __init__(self):
        """
        Initialize the data processor.
        """
        self.scalers = {}
    
    def validate_date_indices(self, *dataframes: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> List[pd.DataFrame]:
        """
        Validate and fix date indices in the provided DataFrames.
        
        This method checks for common datetime index issues and fixes them:
        1. Indices starting at 1970 (Unix epoch) which indicates timestamp conversion issues
        2. Indices with nanosecond precision
        3. RangeIndex mistakenly used instead of DatetimeIndex
        
        Args:
            *dataframes: DataFrames to validate/fix or dictionaries containing DataFrames
            
        Returns:
            List of DataFrames with fixed indices, or dictionaries with fixed DataFrames
        """
        fixed_items = []
        first_valid_dates = []
        last_valid_dates = []
        
        # Function to process a single DataFrame
        def process_df(df, item_name=None):
            if df is None or not isinstance(df, pd.DataFrame):
                return df
                
            df_copy = df.copy()
            
            # Check if this is a dataset that should have a date index
            # Skip conversion if it has a string index with non-date values
            is_date_index = True
            if not isinstance(df_copy.index, pd.DatetimeIndex):
                # Check if the index looks like it contains dates
                if isinstance(df_copy.index, pd.Index) and len(df_copy.index) > 0:
                    # Sample the first value to see if it looks like a date
                    sample_idx = df_copy.index[0]
                    if isinstance(sample_idx, str) and not any(char.isdigit() for char in sample_idx):
                        # If it's a string with no digits, likely not a date
                        is_date_index = False
                        label = f"{item_name}: " if item_name else ""
                        logger.info(f"{label}Identified as static data, skipping alignment")
                        return df_copy
            
            # Convert to DatetimeIndex if not already and it should be a date index
            if not isinstance(df_copy.index, pd.DatetimeIndex) and is_date_index:
                try:
                    df_copy.index = pd.to_datetime(df_copy.index)
                except Exception as e:
                    label = f"{item_name}: " if item_name else ""
                    logger.warning(f"{label}Failed to convert index to DatetimeIndex: {str(e)}")
                    return df_copy
            
            # Store valid date range info for datasets with good dates
            if isinstance(df_copy.index, pd.DatetimeIndex) and len(df_copy) > 0:
                # Only consider dates after 1980 and before 2100 as valid
                good_dates = df_copy.index[(df_copy.index.year > 1980) & (df_copy.index.year < 2100)]
                if len(good_dates) > 0:
                    first_valid_dates.append(good_dates.min())
                    last_valid_dates.append(good_dates.max())
            
            # Check for suspicious dates (Unix epoch) which indicate timestamp conversion issues
            suspicious_dates = (df_copy.index.year == 1970).sum() if isinstance(df_copy.index, pd.DatetimeIndex) else 0
            
            if suspicious_dates > 0:
                label = f"{item_name}: " if item_name else ""
                logger.warning(f"{label}Found {suspicious_dates} suspicious dates from 1970 (Unix epoch)")
                
                # Get the actual start date observed in other datasets or from non-suspicious dates
                actual_start_date = None
                
                # First try to use non-suspicious dates from the current dataset
                if isinstance(df_copy.index, pd.DatetimeIndex):
                    non_suspicious_dates = df_copy.index[df_copy.index.year > 1980]
                    if len(non_suspicious_dates) > 0:
                        actual_start_date = non_suspicious_dates.min()
                        logger.info(f"{label}Using first non-suspicious date from dataset: {actual_start_date}")
                
                # If no valid dates in current dataset, use dates from other datasets
                if actual_start_date is None and first_valid_dates:
                    actual_start_date = min(first_valid_dates)
                    logger.info(f"{label}Using earliest date from other datasets: {actual_start_date}")
                
                # If still no valid date, use a reasonable default for finance data
                if actual_start_date is None:
                    # Find first valid Monday in January 2010 (common market open)
                    actual_start_date = pd.Timestamp('2010-01-04')  # First Monday of January 2010
                    logger.info(f"{label}No valid dates found, defaulting to: {actual_start_date}")
                
                try:
                    # Create a business day date range with the right number of periods
                    corrected_index = pd.date_range(
                        start=actual_start_date, 
                        periods=len(df_copy),
                        freq='B'
                    )
                    
                    df_copy.index = corrected_index
                    logger.info(f"{label}Reconstructed date index from {corrected_index[0]} to {corrected_index[-1]}")
                except Exception as e:
                    logger.error(f"{label}Failed to reconstruct date index: {str(e)}")
            
            return df_copy
        
        # Process each item - could be a DataFrame or a dictionary of DataFrames
        for i, item in enumerate(dataframes):
            if isinstance(item, dict):
                # Process each DataFrame in the dictionary
                fixed_dict = {}
                for key, df in item.items():
                    fixed_dict[key] = process_df(df, item_name=key)
                fixed_items.append(fixed_dict)
            else:
                # Process single DataFrame
                fixed_items.append(process_df(item, item_name=f"DataFrame {i}"))
        
        # Determine consensus date range from all valid data
        if first_valid_dates and last_valid_dates:
            consensus_start = min(first_valid_dates)
            consensus_end = max(last_valid_dates)
            logger.info(f"Consensus date range across datasets: {consensus_start} to {consensus_end}")
            
            # Second pass: Fix any remaining dataframes that weren't fixed in first pass
            for i, item in enumerate(fixed_items):
                if isinstance(item, dict):
                    # Process each DataFrame in the dictionary
                    for key, df in item.items():
                        if not isinstance(df, pd.DataFrame):
                            continue
                            
                        # Skip if already fixed or if it's not a date index
                        if isinstance(df.index, pd.DatetimeIndex) and df.index.min().year > 1980:
                            continue
                        
                        # Skip non-date indices (like ticker symbols)
                        if (isinstance(df.index, pd.Index) and len(df.index) > 0 and 
                            isinstance(df.index[0], str) and not any(char.isdigit() for char in df.index[0])):
                            continue
                            
                        # Check if the dates are still suspicious
                        suspicious = False
                        if isinstance(df.index, pd.DatetimeIndex):
                            suspicious = (df.index.year < 1980).any() or (df.index.year > 2100).any()
                        
                        if suspicious:
                            logger.warning(f"{key}: Still has suspicious dates, using consensus date range")
                            
                            try:
                                # Create a business day date range using consensus dates
                                corrected_index = pd.date_range(
                                    start=consensus_start,
                                    end=consensus_end,
                                    periods=len(df),
                                    freq='B'
                                )
                                
                                item[key].index = corrected_index
                                logger.info(f"{key}: Aligned to consensus date range {corrected_index[0]} to {corrected_index[-1]}")
                            except Exception as e:
                                logger.error(f"{key}: Failed to create corrected index: {str(e)}")
                                # Fallback to a business day range from actual start
                                try:
                                    corrected_index = pd.date_range(
                                        start=consensus_start, 
                                        periods=len(df),
                                        freq='B'
                                    )
                                    item[key].index = corrected_index
                                except:
                                    logger.error(f"{key}: Failed to create fallback index")
                else:
                    # Process single DataFrame
                    df = item
                    if not isinstance(df, pd.DataFrame):
                        continue
                        
                    # Skip if already fixed
                    if isinstance(df.index, pd.DatetimeIndex) and df.index.min().year > 1980:
                        continue
                    
                    # Skip non-date indices (like ticker symbols)
                    if (isinstance(df.index, pd.Index) and len(df.index) > 0 and 
                        isinstance(df.index[0], str) and not any(char.isdigit() for char in df.index[0])):
                        continue
                        
                    # Check if the dates are still suspicious
                    suspicious = False
                    if isinstance(df.index, pd.DatetimeIndex):
                        suspicious = (df.index.year < 1980).any() or (df.index.year > 2100).any()
                    
                    if suspicious:
                        logger.warning(f"DataFrame {i}: Still has suspicious dates, using consensus date range")
                        
                        try:
                            # Create a business day date range with the right number of periods
                            corrected_index = pd.date_range(
                                start=consensus_start,
                                end=consensus_end,
                                periods=len(df),
                                freq='B'
                            )
                            
                            fixed_items[i].index = corrected_index
                            logger.info(f"DataFrame {i}: Aligned to consensus date range {corrected_index[0]} to {corrected_index[-1]}")
                        except Exception as e:
                            logger.error(f"DataFrame {i}: Failed to create corrected index: {str(e)}")
                            # Fallback to a business day range from actual start
                            try:
                                corrected_index = pd.date_range(
                                    start=consensus_start, 
                                    periods=len(df),
                                    freq='B'
                                )
                                fixed_items[i].index = corrected_index
                            except:
                                logger.error(f"DataFrame {i}: Failed to create fallback index")
        
        # If only one DataFrame was passed, return it directly
        if len(dataframes) == 1 and not isinstance(dataframes[0], dict):
            return fixed_items[0]
        
        return fixed_items

    def align_datasets(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Align all datasets to a common date index.
        
        Args:
            data_dict: Dictionary of DataFrames with different datasets
            
        Returns:
            Dictionary of aligned DataFrames
        """
        logger.info("Aligning datasets to a common date index")
        
        # Validate date indices before alignment
        data_dict = self.validate_date_indices(data_dict)[0]
        
        # Separate time series and static data
        time_series_data = {}
        static_data = {}
        
        for df_name, df in data_dict.items():
            # Skip empty DataFrames
            if df is None or df.empty:
                continue
            
            # Check if this is a time series dataset
            is_time_series = False
            
            # If it's already a DatetimeIndex, it's a time series
            if isinstance(df.index, pd.DatetimeIndex):
                is_time_series = True
            else:
                # Check the first few index values to determine if it looks like a date
                if len(df.index) > 0:
                    # Get the first index value
                    sample_idx = df.index[0]
                    
                    # If it's a string with no digits, likely not a date (e.g., ticker symbols)
                    if isinstance(sample_idx, str) and not any(char.isdigit() for char in sample_idx):
                        is_time_series = False
                    else:
                        # Try to convert to datetime as a final check
                        try:
                            pd.to_datetime(sample_idx)
                            is_time_series = True
                        except (ValueError, TypeError):
                            is_time_series = False
                else:
                    is_time_series = False
            
            if is_time_series:
                # Ensure index is DatetimeIndex
                if not isinstance(df.index, pd.DatetimeIndex):
                    try:
                        df = df.copy()
                        df.index = pd.to_datetime(df.index)
                    except Exception as e:
                        logger.warning(f"Failed to convert index to DatetimeIndex for {df_name}: {str(e)}")
                
                time_series_data[df_name] = df
            else:
                static_data[df_name] = df
                logger.info(f"Dataset {df_name} identified as static data, skipping alignment")
        
        # Process time series data
        if time_series_data:
            # Get all unique dates from all time series datasets
            all_dates = set()
            date_info = {}
            
            for df_name, df in time_series_data.items():
                if isinstance(df.index, pd.DatetimeIndex):
                    # Only consider valid dates (after 1980)
                    valid_dates = df.index[df.index.year > 1980]
                    if len(valid_dates) > 0:
                        dates = set(valid_dates)
                        all_dates.update(dates)
                        date_info[df_name] = {
                            'start': valid_dates.min(),
                            'end': valid_dates.max(),
                            'count': len(dates)
                        }
                        logger.info(f"Dataset {df_name} date range: {valid_dates.min()} to {valid_dates.max()} ({len(dates)} dates)")
            
            if all_dates:
                # Sort all unique dates
                all_dates = sorted(all_dates)
                
                # Generate aligned date range (business days) covering all datasets
                start_date = min(df_info['start'] for df_info in date_info.values())
                end_date = max(df_info['end'] for df_info in date_info.values())
                
                # Get intersection of dates across all datasets if needed
                date_sets = [set(df.index[df.index.year > 1980]) for df in time_series_data.values() 
                           if isinstance(df.index, pd.DatetimeIndex) and len(df.index[df.index.year > 1980]) > 0]
                
                if date_sets:
                    common_dates = set.intersection(*date_sets) if len(date_sets) > 1 else date_sets[0]
                    
                    if len(common_dates) < 10:
                        # Use business day range instead if too few common dates
                        logger.warning(f"Only {len(common_dates)} common dates across datasets, using business day range")
                        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
                    else:
                        # Use the common dates as our date range
                        date_range = pd.DatetimeIndex(sorted(common_dates))
                        logger.info(f"Using {len(date_range)} common dates across datasets for alignment")
                else:
                    # Fallback to business day range
                    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
                    logger.info(f"Using business day range from {start_date} to {end_date} for alignment")
                
                # Reindex all time series dataframes to the common date range
                aligned_data = {}
                for df_name, df in time_series_data.items():
                    try:
                        if not isinstance(df.index, pd.DatetimeIndex):
                            df.index = pd.to_datetime(df.index)
                        
                        # Reindex to common date range
                        aligned_df = df.reindex(date_range)
                        
                        # Forward fill missing values (most financial data is carried forward)
                        aligned_df = aligned_df.ffill()
                        
                        # Backward fill any remaining NaNs at the beginning
                        aligned_df = aligned_df.bfill()
                        
                        aligned_data[df_name] = aligned_df
                        
                    except Exception as e:
                        logger.error(f"Error aligning dataset {df_name}: {e}")
                        aligned_data[df_name] = df
                
                # Add static data back
                aligned_data.update(static_data)
                
                # Validate date indices after alignment
                aligned_data = self.validate_date_indices(aligned_data)[0]
                
                logger.info(f"Successfully aligned {len(time_series_data)} time series datasets to {len(date_range)} dates")
                return aligned_data
            else:
                # No valid dates found in any dataset
                logger.warning("No valid dates found in any dataset, returning original data")
                return data_dict
        
        # If no time series data, return original dict
        return data_dict
    
    def compute_derived_features(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Compute derived features from the raw data.
        
        Args:
            data_dict: Dictionary of DataFrames with different datasets
            
        Returns:
            Dictionary of DataFrames with additional derived features
        """
        logger.info("Computing derived features")
        
        # Validate date indices before computation
        data_dict = self.validate_date_indices(data_dict)[0]
        
        derived_data = data_dict.copy()
        
        # 1. Compute yield curve features
        if 'yields' in derived_data:
            yields_df = derived_data['yields'].copy()
            
            # Compute slopes for different segments of the yield curve
            if all(x in yields_df.columns for x in ['3M', '2Y']):
                yields_df['short_slope'] = yields_df['2Y'] - yields_df['3M']
                
            if all(x in yields_df.columns for x in ['2Y', '10Y']):
                yields_df['mid_slope'] = yields_df['10Y'] - yields_df['2Y']
                
            if all(x in yields_df.columns for x in ['10Y', '30Y']):
                yields_df['long_slope'] = yields_df['30Y'] - yields_df['10Y']
            
            # Compute rolling changes
            for col in [c for c in yields_df.columns if c.endswith('Y') or c.endswith('M')]:
                yields_df[f'{col}_1d_change'] = yields_df[col].diff(1)
                yields_df[f'{col}_5d_change'] = yields_df[col].diff(5)
                yields_df[f'{col}_20d_change'] = yields_df[col].diff(20)
            
            # Compute z-scores of slopes (helpful for regime detection)
            for slope in ['short_slope', 'mid_slope', 'long_slope']:
                if slope in yields_df.columns:
                    yields_df[f'{slope}_zscore'] = (
                        yields_df[slope] - yields_df[slope].rolling(252).mean()
                    ) / yields_df[slope].rolling(252).std()
            
            # Compute yield curve curvature
            if all(x in yields_df.columns for x in ['2Y', '5Y', '10Y']):
                yields_df['curvature'] = 2 * yields_df['5Y'] - yields_df['2Y'] - yields_df['10Y']
                yields_df['curvature_zscore'] = (
                    yields_df['curvature'] - yields_df['curvature'].rolling(252).mean()
                ) / yields_df['curvature'].rolling(252).std()
            
            # Detect yield curve inversions
            if '2s10s_spread' in yields_df.columns:
                yields_df['curve_inverted'] = (yields_df['2s10s_spread'] < 0).astype(int)
                # Rolling average of inversion (percentage of days inverted in the last 20 days)
                yields_df['inversion_20d_pct'] = yields_df['curve_inverted'].rolling(20).mean()
            
            derived_data['yields'] = yields_df
        
        # 2. Compute corporate bond features
        if 'corporate' in derived_data:
            corp_df = derived_data['corporate'].copy()
            
            # Compute spreads to Treasury if we have both datasets
            if 'yields' in derived_data and 'AAA' in corp_df.columns and '10Y' in derived_data['yields'].columns:
                corp_df['AAA_10Y_spread'] = corp_df['AAA'] - derived_data['yields']['10Y']
            
            if 'yields' in derived_data and 'BAA' in corp_df.columns and '10Y' in derived_data['yields'].columns:
                corp_df['BAA_10Y_spread'] = corp_df['BAA'] - derived_data['yields']['10Y']
            
            # Compute rolling changes
            for col in corp_df.columns:
                if col not in ['AAA_10Y_spread', 'BAA_10Y_spread']:  # Skip if we just computed these
                    corp_df[f'{col}_1d_change'] = corp_df[col].diff(1)
                    corp_df[f'{col}_5d_change'] = corp_df[col].diff(5)
                    corp_df[f'{col}_20d_change'] = corp_df[col].diff(20)
            
            # Compute z-scores of spreads
            for spread in ['AAA_BAA_spread', 'AAA_10Y_spread', 'BAA_10Y_spread']:
                if spread in corp_df.columns:
                    corp_df[f'{spread}_zscore'] = (
                        corp_df[spread] - corp_df[spread].rolling(252).mean()
                    ) / corp_df[spread].rolling(252).std()
            
            derived_data['corporate'] = corp_df
        
        # Validate date indices after computation
        derived_data = self.validate_date_indices(derived_data)[0]
        
        return derived_data
    
    def prepare_regime_detection_features(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Prepare features specifically for the regime detection model.
        
        Args:
            data_dict: Dictionary of processed DataFrames
            
        Returns:
            DataFrame with features for regime detection, or an empty DataFrame if insufficient data
        """
        logger.info("Preparing features for regime detection")
        
        regime_features = pd.DataFrame()
        
        # Check if we have the minimum required datasets
        required_datasets = ['yields', 'corporate']
        missing_datasets = [dataset for dataset in required_datasets if dataset not in data_dict or data_dict[dataset].empty]
        
        if missing_datasets:
            logger.warning(f"Missing required datasets for regime detection: {missing_datasets}")
            logger.warning("No regime features could be extracted from available data")
            return pd.DataFrame()
        
        # Get a common date index from yields dataset
        if 'yields' in data_dict and isinstance(data_dict['yields'].index, pd.DatetimeIndex):
            date_index = data_dict['yields'].index
            regime_features = pd.DataFrame(index=date_index)
            
            # 1. Yield Curve Features
            yields_df = data_dict['yields']
            
            # Basic yield levels
            key_tenors = ['2Y', '5Y', '10Y', '30Y']
            available_tenors = [tenor for tenor in key_tenors if tenor in yields_df.columns]
            
            if not available_tenors:
                logger.warning("No valid yield tenors found in the data. Required: 2Y, 5Y, 10Y, 30Y")
            
            for tenor in available_tenors:
                regime_features[f'{tenor}_yield'] = yields_df[tenor]
            
            # Yield curve slopes
            if all(x in yields_df.columns for x in ['2Y', '10Y']):
                regime_features['2s10s_spread'] = yields_df['10Y'] - yields_df['2Y']
                # Use a safer approach to calculate z-scores that handles NaNs well
                rolling_mean = regime_features['2s10s_spread'].rolling(252, min_periods=30).mean()
                rolling_std = regime_features['2s10s_spread'].rolling(252, min_periods=30).std()
                # Avoid division by zero or very small values
                rolling_std = rolling_std.replace(0, np.nan)
                min_std = rolling_std.dropna().quantile(0.05) if not rolling_std.dropna().empty else 0.001
                rolling_std = rolling_std.fillna(min_std).replace(0, min_std)
                regime_features['2s10s_zscore'] = (regime_features['2s10s_spread'] - rolling_mean) / rolling_std
                # Fill remaining NaNs with 0 (neutral z-score)
                regime_features['2s10s_zscore'] = regime_features['2s10s_zscore'].fillna(0)
            else:
                logger.warning("Missing 2Y or 10Y yield data for 2s10s spread calculation")
            
            if all(x in yields_df.columns for x in ['3M', '10Y']):
                regime_features['3m10y_spread'] = yields_df['10Y'] - yields_df['3M']
            
            # Yield curve shape
            if all(x in yields_df.columns for x in ['2Y', '5Y', '10Y']):
                regime_features['curve_level'] = (yields_df['2Y'] + yields_df['5Y'] + yields_df['10Y']) / 3
                regime_features['curve_slope'] = yields_df['10Y'] - yields_df['2Y']
                regime_features['curve_curvature'] = 2 * yields_df['5Y'] - yields_df['2Y'] - yields_df['10Y']
            
            # Yield momentum - with NaN handling
            for tenor in available_tenors:
                # Use shorter lookback periods to reduce NaNs
                regime_features[f'{tenor}_1m_change'] = yields_df[tenor].diff(20).fillna(0)
                regime_features[f'{tenor}_3m_change'] = yields_df[tenor].diff(60).fillna(0)
            
            # Yield volatility - with better NaN handling
            for tenor in available_tenors:
                # Use a smaller window and require fewer observations
                regime_features[f'{tenor}_volatility'] = yields_df[tenor].rolling(20, min_periods=5).std() * np.sqrt(252)
                # Fill remaining NaNs with median volatility
                median_vol = regime_features[f'{tenor}_volatility'].dropna().median()
                regime_features[f'{tenor}_volatility'] = regime_features[f'{tenor}_volatility'].fillna(median_vol)
        else:
            logger.warning("No valid yield data found or date index issues")
            return pd.DataFrame()
        
        # 2. Credit Spread Features
        if 'corporate' in data_dict and not data_dict['corporate'].empty:
            corp_df = data_dict['corporate']
            
            # Basic spread levels
            if 'AAA_BAA_spread' in corp_df.columns:
                regime_features['credit_spread'] = corp_df['AAA_BAA_spread']
                
                # Use a safer approach for z-scores
                rolling_mean = corp_df['AAA_BAA_spread'].rolling(252, min_periods=30).mean()
                rolling_std = corp_df['AAA_BAA_spread'].rolling(252, min_periods=30).std()
                # Avoid division by zero or very small values
                rolling_std = rolling_std.replace(0, np.nan)
                min_std = rolling_std.dropna().quantile(0.05) if not rolling_std.dropna().empty else 0.001
                rolling_std = rolling_std.fillna(min_std).replace(0, min_std)
                regime_features['credit_spread_zscore'] = (corp_df['AAA_BAA_spread'] - rolling_mean) / rolling_std
                # Fill remaining NaNs with 0 (neutral z-score)
                regime_features['credit_spread_zscore'] = regime_features['credit_spread_zscore'].fillna(0)
            
                # Spread changes - with NaN handling
                regime_features['credit_spread_1m_change'] = corp_df['AAA_BAA_spread'].diff(20).fillna(0)
                regime_features['credit_spread_3m_change'] = corp_df['AAA_BAA_spread'].diff(60).fillna(0)
            
                # Spread volatility - with better NaN handling
                regime_features['credit_spread_volatility'] = corp_df['AAA_BAA_spread'].rolling(20, min_periods=5).std() * np.sqrt(252)
                # Fill remaining NaNs with median volatility
                median_vol = regime_features['credit_spread_volatility'].dropna().median()
                regime_features['credit_spread_volatility'] = regime_features['credit_spread_volatility'].fillna(median_vol)
            else:
                logger.warning("AAA_BAA_spread column not found in corporate data")
        
        # 3. Market Stress Indicators
        if 'macro' in data_dict and not data_dict['macro'].empty:
            macro_df = data_dict['macro']
            
            # VIX
            if 'VIX' in macro_df.columns:
                regime_features['vix'] = macro_df['VIX']
                
                # Use a safer approach for z-scores
                rolling_mean = macro_df['VIX'].rolling(252, min_periods=30).mean()
                rolling_std = macro_df['VIX'].rolling(252, min_periods=30).std()
                # Avoid division by zero or very small values
                rolling_std = rolling_std.replace(0, np.nan)
                min_std = rolling_std.dropna().quantile(0.05) if not rolling_std.dropna().empty else 0.001
                rolling_std = rolling_std.fillna(min_std).replace(0, min_std)
                regime_features['vix_zscore'] = (macro_df['VIX'] - rolling_mean) / rolling_std
                # Fill remaining NaNs with 0 (neutral z-score)
                regime_features['vix_zscore'] = regime_features['vix_zscore'].fillna(0)
            
            # Inflation
            if 'Inflation_YoY' in macro_df.columns:
                regime_features['inflation'] = macro_df['Inflation_YoY']
                
                # Use a safer approach for z-scores
                rolling_mean = macro_df['Inflation_YoY'].rolling(252, min_periods=30).mean()
                rolling_std = macro_df['Inflation_YoY'].rolling(252, min_periods=30).std()
                # Avoid division by zero or very small values
                rolling_std = rolling_std.replace(0, np.nan)
                min_std = rolling_std.dropna().quantile(0.05) if not rolling_std.dropna().empty else 0.001
                rolling_std = rolling_std.fillna(min_std).replace(0, min_std)
                regime_features['inflation_zscore'] = (macro_df['Inflation_YoY'] - rolling_mean) / rolling_std
                # Fill remaining NaNs with 0 (neutral z-score)
                regime_features['inflation_zscore'] = regime_features['inflation_zscore'].fillna(0)
        
        # Ensure we retain only complete records by dropping NaN rows
        original_rows = len(regime_features)
        regime_features = regime_features.dropna()
        final_rows = len(regime_features)
        
        if final_rows < original_rows:
            pct_dropped = 100 * (original_rows - final_rows) / original_rows
            logger.info(f"Removed {original_rows - final_rows} rows ({pct_dropped:.2f}%) with NaN values from regime features")
            
        # Check if we managed to extract any features
        if regime_features.empty or regime_features.shape[1] == 0:
            logger.warning("No regime features could be extracted from available data")
            return pd.DataFrame()
            
        # Count number of features extracted
        logger.info(f"Successfully extracted {regime_features.shape[1]} regime detection features")
            
        return regime_features
    
    def prepare_rl_features(self, data_dict: Dict[str, pd.DataFrame], n_lags: int = 10) -> pd.DataFrame:
        """
        Prepare features for the RL agent, including lagged features.
        
        Args:
            data_dict: Dictionary of processed DataFrames
            n_lags: Number of lags to include for time series features
            
        Returns:
            DataFrame with features for the RL agent, or an empty DataFrame if insufficient data
        """
        logger.info(f"Preparing features for RL agent with {n_lags} lags")
        
        # Check if we have the minimum required datasets
        required_datasets = ['yields', 'corporate']
        missing_datasets = [dataset for dataset in required_datasets if dataset not in data_dict or data_dict[dataset].empty]
        
        if missing_datasets:
            logger.warning(f"Missing required datasets for RL features: {missing_datasets}")
            logger.warning("No RL features could be extracted from available data")
            return pd.DataFrame()
        
        # Start with regime detection features as our base
        rl_features = self.prepare_regime_detection_features(data_dict)
        
        # If regime features couldn't be extracted, we can't proceed
        if rl_features.empty:
            logger.warning("No regime detection features available, can't prepare RL features")
            return pd.DataFrame()
        
        # Add more specific features for the RL agent
        
        # 1. Add ETF returns as asset class performance indicators
        if 'etf_features' in data_dict and not data_dict['etf_features'].empty:
            etf_df = data_dict['etf_features']
            
            # Daily returns
            return_features = [col for col in etf_df.columns if 'avg_return' in col and not '20d' in col]
            if return_features:
                for feature in return_features:
                    # Fill NaN values with 0 (no return)
                    rl_features[feature] = etf_df[feature].reindex(rl_features.index).fillna(0)
            else:
                logger.warning("No ETF return features found")
            
            # Volatility
            vol_features = [col for col in etf_df.columns if 'volatility' in col]
            if vol_features:
                for feature in vol_features:
                    # Fill NaN values with the median for the feature
                    feature_vals = etf_df[feature].reindex(rl_features.index)
                    if not feature_vals.empty and not feature_vals.dropna().empty:
                        median_val = feature_vals.dropna().median()
                        rl_features[feature] = feature_vals.fillna(median_val)
                    else:
                        rl_features[feature] = feature_vals.fillna(0)
            else:
                logger.warning("No ETF volatility features found")
        
        # 2. Add daily yield changes
        if 'yields' in data_dict and not data_dict['yields'].empty:
            yields_df = data_dict['yields']
            
            # Daily changes for key tenors
            key_tenors = ['2Y', '5Y', '10Y', '30Y']
            available_daily_changes = [tenor for tenor in key_tenors if f'{tenor}_1d_change' in yields_df.columns]
            
            if available_daily_changes:
                for tenor in available_daily_changes:
                    rl_features[f'{tenor}_daily_change'] = yields_df[f'{tenor}_1d_change'].reindex(rl_features.index).fillna(0)
            else:
                # Try to compute daily changes if they don't exist
                available_tenors = [tenor for tenor in key_tenors if tenor in yields_df.columns]
                if available_tenors:
                    for tenor in available_tenors:
                        rl_features[f'{tenor}_daily_change'] = yields_df[tenor].reindex(rl_features.index).diff().fillna(0)
                    logger.info(f"Computed daily changes for {len(available_tenors)} yield tenors")
                else:
                    logger.warning("No yield tenor data available for daily changes")
        
        # 3. Add daily credit spread changes
        if 'corporate' in data_dict and not data_dict['corporate'].empty:
            corp_df = data_dict['corporate']
            
            # Daily changes for key spreads
            key_spreads = ['AAA_BAA_spread', 'AAA_10Y_spread', 'BAA_10Y_spread']
            available_daily_changes = [spread for spread in key_spreads if f'{spread}_1d_change' in corp_df.columns]
            
            if available_daily_changes:
                for spread in available_daily_changes:
                    rl_features[f'{spread}_daily_change'] = corp_df[f'{spread}_1d_change'].reindex(rl_features.index).fillna(0)
            else:
                # Try to compute daily changes if they don't exist
                available_spreads = [spread for spread in key_spreads if spread in corp_df.columns]
                if available_spreads:
                    for spread in available_spreads:
                        rl_features[f'{spread}_daily_change'] = corp_df[spread].reindex(rl_features.index).diff().fillna(0)
                    logger.info(f"Computed daily changes for {len(available_spreads)} credit spreads")
                else:
                    logger.warning("No credit spread data available for daily changes")
        
        # Check if we have any features before generating lags
        if rl_features.empty or rl_features.shape[1] == 0:
            logger.warning("No RL features could be extracted from available data")
            return pd.DataFrame()
        
        # Save original index for verification later
        original_index = rl_features.index.copy()
        
        # Create lagged features for time series
        if n_lags > 0:
            original_cols = rl_features.columns.tolist()
            
            # Initialize a list to hold all lag DataFrames
            lag_dfs = []
            
            for col in original_cols:
                # Skip categorical features
                if rl_features[col].dtype == 'object' or rl_features[col].dtype == 'category':
                    continue
                
                # Create a DataFrame with all lags for this column
                col_lags = pd.DataFrame(
                    {f'{col}_lag{lag}': rl_features[col].shift(lag) 
                     for lag in range(1, n_lags + 1)},
                    index=rl_features.index
                )
                
                lag_dfs.append(col_lags)
            
            # Concatenate all lag DataFrames horizontally
            if lag_dfs:
                all_lags = pd.concat(lag_dfs, axis=1)
                
                # Add the lagged features to the original features
                rl_features = pd.concat([rl_features, all_lags], axis=1)
            else:
                logger.warning("No numeric features available for creating lags")
        
        # Drop rows with missing values due to lags
        # For RL features, we need to ensure we have valid data (not NaNs from lag creation)
        valid_rows_before = len(rl_features)
        rl_features = rl_features.dropna()
        valid_rows_after = len(rl_features)
        
        if valid_rows_after < valid_rows_before:
            dropped_pct = 100 * (valid_rows_before - valid_rows_after) / valid_rows_before
            logger.info(f"Dropped {valid_rows_before - valid_rows_after} rows ({dropped_pct:.2f}%) with missing values due to lag creation")
        
        # Verify we still have a contiguous date index after dropping rows
        if len(rl_features) > 0:
            # Check if we dropped all initial n_lags values as expected
            expected_start_idx = n_lags
            if len(original_index) > expected_start_idx:
                expected_start_date = original_index[expected_start_idx]
                actual_start_date = rl_features.index[0]
                
                if expected_start_date != actual_start_date:
                    logger.warning(f"Date alignment issue: Expected to start at {expected_start_date} but actually starts at {actual_start_date}")
        
        if rl_features.empty:
            logger.warning("After removing NaN values, no RL features remain")
            return pd.DataFrame()
        
        # Log the number of features
        logger.info(f"Prepared {rl_features.shape[1]} features for RL agent across {len(rl_features)} dates")
        
        return rl_features
    
    def normalize_features(self, features_df: pd.DataFrame, scaler_type: str = 'standard', 
                        fit: bool = True) -> pd.DataFrame:
        """
        Normalize features using StandardScaler or MinMaxScaler.
        
        Args:
            features_df: DataFrame with features to normalize
            scaler_type: Type of scaling ('standard' or 'minmax')
            fit: Whether to fit a new scaler or use a previously fitted one
            
        Returns:
            DataFrame with normalized features
        """
        logger.info(f"Normalizing features using {scaler_type} scaling")
        
        # Safety check for empty dataframe
        if features_df.empty or len(features_df.columns) == 0:
            logger.warning("Empty DataFrame provided for normalization. Returning empty DataFrame.")
            return pd.DataFrame(index=features_df.index)
        
        # Create a copy to avoid modifying the original
        normalized_df = features_df.copy()
        
        # Select the appropriate scaler
        if fit:
            if scaler_type == 'standard':
                scaler = StandardScaler()
            elif scaler_type == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaler type: {scaler_type}")
            
            # Save the scaler for later use
            self.scalers[scaler_type] = scaler
        else:
            # Use a previously fitted scaler
            if scaler_type not in self.scalers:
                raise ValueError(f"No fitted scaler of type {scaler_type} found. Run with fit=True first.")
            
            scaler = self.scalers[scaler_type]
        
        # Get numerical columns
        num_cols = normalized_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Safety check for empty numerical columns
        if not num_cols:
            logger.warning("No numerical columns found for normalization. Returning original DataFrame.")
            return normalized_df
        
        if fit:
            # Fit and transform
            normalized_values = scaler.fit_transform(normalized_df[num_cols])
        else:
            # Transform only
            normalized_values = scaler.transform(normalized_df[num_cols])
        
        # Replace values in the DataFrame
        normalized_df[num_cols] = normalized_values
        
        logger.info(f"Successfully normalized {len(num_cols)} features")
        
        return normalized_df
    
    def split_time_series(self, data: pd.DataFrame, train_ratio: float = 0.7, 
                        val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split time series data into train, validation, and test sets.
        
        Args:
            data: DataFrame with time series data
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            
        Returns:
            (train_df, val_df, test_df) tuple of DataFrames
        """
        logger.info(f"Splitting time series data with train_ratio={train_ratio}, val_ratio={val_ratio}")
        
        # Check if DataFrame is empty
        if data is None or data.empty:
            logger.warning("Empty DataFrame provided for splitting. Returning empty DataFrames.")
            empty_df = pd.DataFrame(columns=data.columns if data is not None else [])
            return empty_df, empty_df.copy(), empty_df.copy()
        
        # Check if this is actually time series data with a date index
        is_date_index = True
        if not isinstance(data.index, pd.DatetimeIndex):
            # Check if the index appears to be a date index
            if len(data.index) > 0:
                sample_idx = data.index[0]
                if isinstance(sample_idx, str) and not any(char.isdigit() for char in sample_idx):
                    # If it's a string with no digits, likely not a date (e.g., ticker symbols)
                    is_date_index = False
                    logger.warning("Data appears to have a non-date index. Using positional splitting.")
                    
                    # Fall back to positional splitting
                    n = len(data)
                    train_idx = int(n * train_ratio)
                    val_idx = train_idx + int(n * val_ratio)
                    
                    train_df = data.iloc[:train_idx]
                    val_df = data.iloc[train_idx:val_idx]
                    test_df = data.iloc[val_idx:]
                    
                    return train_df, val_df, test_df
            
            # Try to convert if it still looks like it could be a date
            if is_date_index:
                logger.warning("Data index is not a DatetimeIndex. Trying to convert...")
                try:
                    data.index = pd.to_datetime(data.index)
                except:
                    logger.error("Could not convert index to DatetimeIndex. Using positional splitting instead.")
                    # Fall back to positional splitting
                    n = len(data)
                    train_idx = int(n * train_ratio)
                    val_idx = train_idx + int(n * val_ratio)
                    
                    train_df = data.iloc[:train_idx]
                    val_df = data.iloc[train_idx:val_idx]
                    test_df = data.iloc[val_idx:]
                    
                    return train_df, val_df, test_df
        
        # Sort data by date
        data = data.sort_index()
        
        # Calculate split dates
        start_date = data.index.min()
        end_date = data.index.max()
        date_range = (end_date - start_date).days
        
        train_days = int(date_range * train_ratio)
        val_days = int(date_range * val_ratio)
        
        train_end_date = start_date + pd.DateOffset(days=train_days)
        val_end_date = train_end_date + pd.DateOffset(days=val_days)
        
        # Split the data
        train_df = data.loc[start_date:train_end_date].copy()
        val_df = data.loc[train_end_date:val_end_date].copy()
        test_df = data.loc[val_end_date:].copy()
        
        logger.info(f"Split time series data into train ({len(train_df)}), val ({len(val_df)}), and test ({len(test_df)}) sets")
        return train_df, val_df, test_df
    
    def process_data_pipeline(self, data_dict: Dict[str, pd.DataFrame], 
                            for_regime: bool = True, 
                            for_rl: bool = True,
                            n_lags: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Run the complete data processing pipeline.
        
        Args:
            data_dict: Dictionary of raw DataFrames
            for_regime: Whether to prepare features for regime detection
            for_rl: Whether to prepare features for the RL agent
            n_lags: Number of lags for time series features
            
        Returns:
            Dictionary with processed datasets
        """
        logger.info("Running complete data processing pipeline")
        
        # Validate input data
        data_dict = self.validate_date_indices(data_dict)[0]
        
        # Step 1: Align datasets
        aligned_data = self.align_datasets(data_dict)
        
        # Validate after alignment
        aligned_data = self.validate_date_indices(aligned_data)[0]
        
        # Step 2: Compute derived features
        processed_data = self.compute_derived_features(aligned_data)
        
        # Validate after feature computation
        processed_data = self.validate_date_indices(processed_data)[0]
        
        # Step 3: Drop NaN values from processed yields and propagate consistent dates
        if 'yields' in processed_data and not processed_data['yields'].empty:
            # Log the original data shape
            original_rows = len(processed_data['yields'])
            logger.info(f"Original processed yields data has {original_rows} rows")
            
            # Drop rows with NaN values
            yields_clean = processed_data['yields'].dropna()
            
            # Log the result
            clean_rows = len(yields_clean)
            if clean_rows < original_rows:
                dropped_pct = 100 * (original_rows - clean_rows) / original_rows
                logger.info(f"Dropped {original_rows - clean_rows} rows ({dropped_pct:.2f}%) with NaN values from yields data")
                logger.info(f"Clean yields data starts at {yields_clean.index[0]} and has {clean_rows} rows")
                
                # Update processed yields with clean data
                processed_data['yields'] = yields_clean
                
                # Get the clean dates to filter other datasets for consistency
                clean_dates = yields_clean.index
                
                # Apply the same date filter to all other time series datasets
                for key, df in processed_data.items():
                    if key != 'yields' and isinstance(df, pd.DataFrame) and not df.empty:
                        if isinstance(df.index, pd.DatetimeIndex):
                            # Only keep dates that are in the clean_dates
                            filtered_df = df.loc[df.index.isin(clean_dates)]
                            
                            # Log the filtering
                            if len(filtered_df) < len(df):
                                logger.info(f"Filtered {key} dataset from {len(df)} to {len(filtered_df)} rows to match clean yields dates")
                            
                            # Update the dataset
                            processed_data[key] = filtered_df

        # Prepare output dictionary
        output = {'processed': processed_data}
        
        # Step 4: Prepare regime detection features
        regime_features = None
        if for_regime:
            regime_features = self.prepare_regime_detection_features(processed_data)
            
            # Check if any features were extracted
            if regime_features is None or regime_features.empty:
                logger.warning("No regime detection features could be extracted. Skipping regime-related processing.")
                # Add empty placeholders to maintain expected output structure
                output['regime_features'] = pd.DataFrame()
                output['regime_features_normalized'] = pd.DataFrame()
                output['regime_train'] = pd.DataFrame()
                output['regime_val'] = pd.DataFrame()
                output['regime_test'] = pd.DataFrame()
            else:
                # Drop rows with NaN values from regime features
                valid_rows_before = len(regime_features)
                regime_features = regime_features.dropna()
                valid_rows_after = len(regime_features)
                
                if valid_rows_after < valid_rows_before:
                    dropped_pct = 100 * (valid_rows_before - valid_rows_after) / valid_rows_before
                    logger.info(f"Dropped {valid_rows_before - valid_rows_after} rows ({dropped_pct:.2f}%) with missing values from regime features")
                    logger.info(f"First date with complete regime detection data: {regime_features.index[0]}")
                
                # Validate regime features
                regime_features = self.validate_date_indices(regime_features)
                
                output['regime_features'] = regime_features
                
                # Normalize features
                regime_features_norm = self.normalize_features(regime_features, 'standard')
                
                # Validate normalized features
                regime_features_norm = self.validate_date_indices(regime_features_norm)
                
                output['regime_features_normalized'] = regime_features_norm
                
                # Split data
                train_df, val_df, test_df = self.split_time_series(regime_features_norm)
                
                # Validate split data
                train_df = self.validate_date_indices(train_df)
                val_df = self.validate_date_indices(val_df)
                test_df = self.validate_date_indices(test_df)
                
                output['regime_train'] = train_df
                output['regime_val'] = val_df
                output['regime_test'] = test_df
        
        # Step 5: Prepare RL features
        rl_features = None
        if for_rl:
            rl_features = self.prepare_rl_features(processed_data, n_lags)
            
            # Check if any features were extracted
            if rl_features is None or rl_features.empty:
                logger.warning("No RL features could be extracted. Skipping RL-related processing.")
                # Add empty placeholders to maintain expected output structure
                output['rl_features'] = pd.DataFrame()
                output['rl_features_normalized'] = pd.DataFrame()
                output['rl_train'] = pd.DataFrame()
                output['rl_val'] = pd.DataFrame()
                output['rl_test'] = pd.DataFrame()
            else:
                # Make sure RL features don't have NaN values
                if rl_features.isna().any().any():
                    valid_rows_before = len(rl_features)
                    rl_features = rl_features.dropna()
                    valid_rows_after = len(rl_features)
                    
                    if valid_rows_after < valid_rows_before:
                        dropped_pct = 100 * (valid_rows_before - valid_rows_after) / valid_rows_before
                        logger.info(f"Dropped {valid_rows_before - valid_rows_after} rows ({dropped_pct:.2f}%) with missing values from RL features")
                        logger.info(f"First date with complete RL data: {rl_features.index[0]}")
                
                # Validate RL features
                rl_features = self.validate_date_indices(rl_features)
                
                output['rl_features'] = rl_features
                
                # Normalize features
                rl_features_norm = self.normalize_features(rl_features, 'standard')
                
                # Validate normalized features
                rl_features_norm = self.validate_date_indices(rl_features_norm)
                
                output['rl_features_normalized'] = rl_features_norm
                
                # Split data
                train_df, val_df, test_df = self.split_time_series(rl_features_norm)
                
                # Validate split data
                train_df = self.validate_date_indices(train_df)
                val_df = self.validate_date_indices(val_df)
                test_df = self.validate_date_indices(test_df)
                
                output['rl_train'] = train_df
                output['rl_val'] = val_df
                output['rl_test'] = test_df
        
        # Step 6: Ensure consistent start dates between regime and RL features
        if regime_features is not None and not regime_features.empty and rl_features is not None and not rl_features.empty:
            # Get the start date of each dataset
            regime_start_date = regime_features.index[0]
            rl_start_date = rl_features.index[0]
            
            # Use the later of the two dates as the common start date
            common_start_date = max(regime_start_date, rl_start_date)
            logger.info(f"Aligning both datasets to common start date: {common_start_date}")
            
            # Update regime features if needed
            if regime_start_date < common_start_date:
                # Trim regime features
                logger.info(f"Trimming regime features from {regime_start_date} to {common_start_date}")
                regime_features = regime_features.loc[common_start_date:]
                output['regime_features'] = regime_features
                
                # Trim normalized regime features
                if 'regime_features_normalized' in output and not output['regime_features_normalized'].empty:
                    output['regime_features_normalized'] = output['regime_features_normalized'].loc[common_start_date:]
                
                # Re-split regime features to maintain consistency
                train_df, val_df, test_df = self.split_time_series(output['regime_features_normalized'])
                output['regime_train'] = train_df
                output['regime_val'] = val_df
                output['regime_test'] = test_df
            
            # Update RL features if needed
            if rl_start_date < common_start_date:
                # Trim RL features
                logger.info(f"Trimming RL features from {rl_start_date} to {common_start_date}")
                rl_features = rl_features.loc[common_start_date:]
                output['rl_features'] = rl_features
                
                # Trim normalized RL features
                if 'rl_features_normalized' in output and not output['rl_features_normalized'].empty:
                    output['rl_features_normalized'] = output['rl_features_normalized'].loc[common_start_date:]
                
                # Re-split RL features to maintain consistency
                train_df, val_df, test_df = self.split_time_series(output['rl_features_normalized'])
                output['rl_train'] = train_df
                output['rl_val'] = val_df
                output['rl_test'] = test_df
        
        logger.info("Data processing pipeline completed successfully")
        return output
