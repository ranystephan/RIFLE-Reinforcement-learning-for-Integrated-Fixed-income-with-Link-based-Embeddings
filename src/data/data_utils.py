"""
Data Utilities Module for Fixed Income RL Project

This module provides utility functions for data handling:
1. File I/O operations
2. Date handling functions
3. Matrix operations for yield curve analysis
4. Visualization utilities for data exploration

Author: ranycs & cosrv
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Union, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ensure_dir(directory: str):
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def save_dataframe(df: pd.DataFrame, filename: str, directory: str = 'data/processed'):
    """
    Save a DataFrame to a CSV file.
    
    Args:
        df: DataFrame to save
        filename: Name of the file
        directory: Directory to save the file in
    """
    ensure_dir(directory)
    filepath = os.path.join(directory, filename)
    df.to_csv(filepath)
    logger.info(f"Saved DataFrame to {filepath}")

def load_dataframe(filename: str, directory: str = 'data/processed') -> pd.DataFrame:
    """
    Load a DataFrame from a CSV file.
    
    Args:
        filename: Name of the file
        directory: Directory to load the file from
        
    Returns:
        Loaded DataFrame
    """
    filepath = os.path.join(directory, filename)
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    logger.info(f"Loaded DataFrame from {filepath} with shape {df.shape}")
    return df

def save_pickle(obj, filename: str, directory: str = 'data/processed'):
    """
    Save an object to a pickle file.
    
    Args:
        obj: Object to save
        filename: Name of the file
        directory: Directory to save the file in
    """
    ensure_dir(directory)
    filepath = os.path.join(directory, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    logger.info(f"Saved object to {filepath}")

def load_pickle(filename: str, directory: str = 'data/processed'):
    """
    Load an object from a pickle file.
    
    Args:
        filename: Name of the file
        directory: Directory to load the file from
        
    Returns:
        Loaded object
    """
    filepath = os.path.join(directory, filename)
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    logger.info(f"Loaded object from {filepath}")
    return obj

def get_business_days(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """
    Get a DatetimeIndex of business days between start_date and end_date.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        
    Returns:
        DatetimeIndex of business days
    """
    return pd.date_range(start=start_date, end=end_date, freq='B')

def date_to_str(date: Union[str, datetime, pd.Timestamp]) -> str:
    """
    Convert a date to a string in 'YYYY-MM-DD' format.
    
    Args:
        date: Date to convert
        
    Returns:
        Date string in 'YYYY-MM-DD' format
    """
    if isinstance(date, str):
        return date
    elif isinstance(date, (datetime, pd.Timestamp)):
        return date.strftime('%Y-%m-%d')
    else:
        raise ValueError(f"Unsupported date type: {type(date)}")

def str_to_date(date_str: str) -> datetime:
    """
    Convert a string in 'YYYY-MM-DD' format to a datetime object.
    
    Args:
        date_str: Date string in 'YYYY-MM-DD' format
        
    Returns:
        Datetime object
    """
    return datetime.strptime(date_str, '%Y-%m-%d')

def is_date_range_valid(start_date: str, end_date: str) -> bool:
    """
    Check if a date range is valid.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        
    Returns:
        True if start_date <= end_date, False otherwise
    """
    start = str_to_date(start_date)
    end = str_to_date(end_date)
    return start <= end

def interpolate_yield_curve(tenors: List[float], yields: List[float], 
                          target_tenors: List[float]) -> List[float]:
    """
    Interpolate a yield curve to get yields at target tenors.
    
    Args:
        tenors: List of tenors (in years) for which yields are known
        yields: List of yields corresponding to tenors
        target_tenors: List of tenors to interpolate yields for
        
    Returns:
        List of interpolated yields
    """
    from scipy.interpolate import interp1d
    
    # Create an interpolation function
    interpolator = interp1d(tenors, yields, kind='cubic', bounds_error=False, fill_value='extrapolate')
    
    # Interpolate yields at target tenors
    interpolated_yields = interpolator(target_tenors)
    
    return interpolated_yields

def calculate_duration(cash_flows: List[float], times: List[float], yield_rate: float) -> float:
    """
    Calculate the Macaulay duration of a bond.
    
    Args:
        cash_flows: List of cash flows
        times: List of times to cash flows (in years)
        yield_rate: Yield rate (decimal)
        
    Returns:
        Macaulay duration
    """
    # Calculate present value of each cash flow
    pv_factors = [cf / ((1 + yield_rate) ** t) for cf, t in zip(cash_flows, times)]
    pv = sum(pv_factors)
    
    # Calculate weighted average time
    weighted_times = [pv_factor * t / pv for pv_factor, t in zip(pv_factors, times)]
    
    # Duration is the sum of weighted times
    duration = sum(weighted_times)
    
    return duration

def calculate_modified_duration(duration: float, yield_rate: float) -> float:
    """
    Calculate the modified duration of a bond.
    
    Args:
        duration: Macaulay duration
        yield_rate: Yield rate (decimal)
        
    Returns:
        Modified duration
    """
    return duration / (1 + yield_rate)

def calculate_convexity(cash_flows: List[float], times: List[float], yield_rate: float) -> float:
    """
    Calculate the convexity of a bond.
    
    Args:
        cash_flows: List of cash flows
        times: List of times to cash flows (in years)
        yield_rate: Yield rate (decimal)
        
    Returns:
        Convexity
    """
    # Calculate present value of each cash flow
    pv_factors = [cf / ((1 + yield_rate) ** t) for cf, t in zip(cash_flows, times)]
    pv = sum(pv_factors)
    
    # Calculate weighted average squared time
    weighted_times_squared = [pv_factor * t * (t + 1) / pv / (1 + yield_rate) ** 2 
                             for pv_factor, t in zip(pv_factors, times)]
    
    # Convexity is the sum of weighted times squared
    convexity = sum(weighted_times_squared)
    
    return convexity

def decompose_yield_curve(yields_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Decompose yield curve data into principal components.
    
    Args:
        yields_df: DataFrame with yield curve data (columns are tenors)
        
    Returns:
        Tuple of (components_df, explained_variance)
    """
    from sklearn.decomposition import PCA
    
    # Fill missing values
    filled_df = yields_df.ffill().bfill()
    
    # Standardize data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(filled_df)
    
    # Perform PCA
    pca = PCA(n_components=3)  # Usually 3 components (level, slope, curvature)
    components = pca.fit_transform(scaled_data)
    
    # Create DataFrame of components
    components_df = pd.DataFrame(
        components, 
        index=filled_df.index, 
        columns=['PC1_Level', 'PC2_Slope', 'PC3_Curvature']
    )
    
    # Get explained variance
    explained_variance = pd.Series(
        pca.explained_variance_ratio_,
        index=['PC1_Level', 'PC2_Slope', 'PC3_Curvature']
    )
    
    return components_df, explained_variance

def plot_yield_curve(yields: List[float], tenors: List[float], 
                   title: str = 'Yield Curve', 
                   date: Optional[str] = None,
                   ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot a yield curve.
    
    Args:
        yields: List of yields
        tenors: List of tenors (in years)
        title: Plot title
        date: Date string for the yield curve
        ax: Matplotlib axes to plot on
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot yield curve
    ax.plot(tenors, yields, 'o-', linewidth=2)
    
    # Set axis labels and title
    ax.set_xlabel('Tenor (years)')
    ax.set_ylabel('Yield (%)')
    
    if date:
        title = f"{title} - {date}"
    
    ax.set_title(title)
    ax.grid(True)
    
    return ax

def plot_yield_curve_3d(yields_df: pd.DataFrame, tenors: List[float], 
                      title: str = '3D Yield Curve Evolution') -> plt.Figure:
    """
    Create a 3D plot of yield curve evolution over time.
    
    Args:
        yields_df: DataFrame with yield curve data (index is dates, columns are tenors)
        tenors: List of tenors (in years)
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # Sample dates (we'll use a subset to avoid overcrowding)
    dates = yields_df.index
    n_dates = len(dates)
    step = max(1, n_dates // 50)  # Sample at most 50 dates
    sampled_dates = dates[::step]
    
    # Create meshgrid
    X, Y = np.meshgrid(tenors, range(len(sampled_dates)))
    
    # Get yields data
    Z = yields_df.loc[sampled_dates].values
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create surface plot
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
    
    # Set axis labels and title
    ax.set_xlabel('Tenor (years)')
    ax.set_ylabel('Time')
    ax.set_zlabel('Yield (%)')
    
    # Set y-tick labels to dates
    from matplotlib.ticker import FixedLocator, FormatStrFormatter
    ax.yaxis.set_major_locator(FixedLocator(range(0, len(sampled_dates), 5)))
    date_labels = [d.strftime('%Y-%m-%d') for d in sampled_dates[::5]]
    ax.set_yticklabels(date_labels)
    
    ax.set_title(title)
    
    # Add color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    return fig

def plot_regime_detection(regime_df: pd.DataFrame) -> plt.Figure:
    """
    Plot the results of regime detection.
    
    Args:
        regime_df: DataFrame with regime labels (index is dates, 'regime' column)
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get regime labels and colors
    regimes = regime_df['regime'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(regimes)))
    
    # Plot background colors for each regime
    for i, regime in enumerate(regimes):
        regime_dates = regime_df[regime_df['regime'] == regime].index
        for j in range(len(regime_dates) - 1):
            ax.axvspan(regime_dates[j], regime_dates[j+1], alpha=0.3, color=colors[i])
    
    # Plot regime labels at the top
    for i, regime in enumerate(regimes):
        regime_periods = []
        regime_dates = regime_df[regime_df['regime'] == regime].index
        if len(regime_dates) > 0:
            start_date = regime_dates[0]
            prev_date = regime_dates[0]
            
            for date in regime_dates[1:]:
                if (date - prev_date).days > 5:  # Gap in regime
                    regime_periods.append((start_date, prev_date))
                    start_date = date
                prev_date = date
            
            regime_periods.append((start_date, regime_dates[-1]))
            
            # Add text for each regime period
            for start, end in regime_periods:
                mid_point = start + (end - start) / 2
                ax.text(mid_point, 1.01, regime, transform=ax.get_xaxis_transform(),
                       ha='center', va='bottom', fontsize=10, color=colors[i])
    
    # Add other time series data if available
    if 'yields_10Y' in regime_df.columns:
        ax.plot(regime_df.index, regime_df['yields_10Y'], 'k-', label='10Y Yield')
    
    if '2s10s_spread' in regime_df.columns:
        ax.plot(regime_df.index, regime_df['2s10s_spread'], 'r-', label='2s10s Spread')
    
    ax.set_xlabel('Date')
    ax.legend()
    ax.set_title('Interest Rate Regimes')
    
    return fig

def generate_synthetic_market_data(start_date: Union[str, datetime], 
                                 num_days: int, 
                                 add_noise: bool = True, 
                                 random_seed: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """
    Generate synthetic market data for testing the pipeline.
    
    Args:
        start_date: Start date for the synthetic data
        num_days: Number of days to generate
        add_noise: Whether to add random noise to the data
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with synthetic market data
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Convert start_date to datetime if it's a string
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    
    # Create date range
    date_range = pd.date_range(start=start_date, periods=num_days, freq='B')
    
    # Create treasury yield curves
    yields_df = pd.DataFrame(index=date_range)
    
    # Base rates for each tenor
    base_rates = {
        '3M': 0.5, '6M': 0.8, '1Y': 1.2, 
        '2Y': 1.8, '3Y': 2.1, '5Y': 2.5, 
        '7Y': 2.8, '10Y': 3.0, '20Y': 3.3, '30Y': 3.5
    }
    
    # Generate yields with mean-reverting properties and regime changes
    n_days = len(date_range)
    
    # Create regimes
    n_regimes = 4
    regime_lengths = np.random.randint(40, 120, size=n_regimes)
    total_length = sum(regime_lengths)
    if total_length < n_days:
        regime_lengths = np.append(regime_lengths, [n_days - total_length])
    
    # Make sure we have enough regime values (repeat if needed)
    regimes = []
    remaining_days = n_days
    regime_idx = 0
    
    while remaining_days > 0:
        if regime_idx >= len(regime_lengths):
            regime_idx = 0  # Cycle back to the first regime
            
        current_length = min(regime_lengths[regime_idx], remaining_days)
        regimes.extend([regime_idx] * current_length)
        
        remaining_days -= current_length
        regime_idx += 1
    
    # Ensure regimes is exactly n_days long
    regimes = regimes[:n_days]
    
    # Parameters for each regime
    regime_params = {
        0: {'mean_shift': 0.0, 'vol': 0.05, 'mean_reversion': 0.05},  # Normal
        1: {'mean_shift': 0.5, 'vol': 0.08, 'mean_reversion': 0.03},  # Rising rates
        2: {'mean_shift': -0.3, 'vol': 0.12, 'mean_reversion': 0.01},  # Falling rates
        3: {'mean_shift': 0.8, 'vol': 0.2, 'mean_reversion': 0.02}     # Stressed
    }
    
    # Make sure all regimes are within valid range (0-3)
    regimes = [r % n_regimes for r in regimes]
    
    # Generate yield curves
    for tenor, base_rate in base_rates.items():
        # Start with base rate
        rates = np.zeros(n_days)
        rates[0] = base_rate
        
        # Add random walk with mean reversion
        for i in range(1, n_days):
            regime = int(regimes[i]) if i < len(regimes) else 0
            params = regime_params[regime]
            
            # Mean-reverting random walk
            target = base_rate + params['mean_shift']
            mean_reversion = params['mean_reversion'] * (target - rates[i-1])
            random_shock = np.random.normal(0, params['vol'])
            
            rates[i] = rates[i-1] + mean_reversion + random_shock
            
        # Ensure rates are positive
        rates = np.maximum(0.01, rates)
        
        # Add to DataFrame
        yields_df[tenor] = rates
    
    # Add derived yield curve measures
    yields_df['2s10s_spread'] = yields_df['10Y'] - yields_df['2Y']
    yields_df['3m10y_spread'] = yields_df['10Y'] - yields_df['3M']
    yields_df['curvature'] = 2 * yields_df['5Y'] - yields_df['2Y'] - yields_df['10Y']
    yields_df['level'] = yields_df[['2Y', '5Y', '10Y']].mean(axis=1)
    yields_df['slope'] = yields_df['2s10s_spread']
    
    # Create corporate bond data
    corporate_df = pd.DataFrame(index=date_range)
    
    # Base spreads for each rating
    base_spreads = {
        'AAA': 0.3, 'AA': 0.5, 'A': 0.8, 
        'BBB': 1.5, 'BB': 3.0, 'B': 5.0, 'CCC': 8.0
    }
    
    # Generate spreads with correlation to treasury yields
    for rating, base_spread in base_spreads.items():
        # Start with base spread
        spreads = np.zeros(n_days)
        spreads[0] = base_spread
        
        # Add random walk with correlation to treasury
        for i in range(1, n_days):
            regime = int(regimes[i]) if i < len(regimes) else 0
            params = regime_params[regime]
            
            # Higher volatility for lower ratings
            rating_vol_multiplier = 1.0
            if rating in ['BB', 'B', 'CCC']:
                rating_vol_multiplier = 1.5
            elif rating == 'BBB':
                rating_vol_multiplier = 1.2
            
            # Mean-reverting random walk with correlation to treasury moves
            treasury_change = yields_df['10Y'].iloc[i] - yields_df['10Y'].iloc[i-1]
            target = base_spread * (1 + params['mean_shift'] * 0.5)
            mean_reversion = 0.1 * (target - spreads[i-1])
            treasury_effect = -0.5 * treasury_change  # Negative correlation with treasury
            random_shock = np.random.normal(0, params['vol'] * rating_vol_multiplier)
            
            spreads[i] = spreads[i-1] + mean_reversion + treasury_effect + random_shock
            
        # Ensure spreads are positive
        spreads = np.maximum(0.01, spreads)
        
        # Add to DataFrame
        corporate_df[rating] = spreads
    
    # Add spread measures
    corporate_df['AAA_BAA_spread'] = corporate_df['BBB'] - corporate_df['AAA']
    corporate_df['BBB_HY_spread'] = corporate_df['BB'] - corporate_df['BBB']
    
    # Add regime column to yields_df
    yields_df['regime'] = regimes
    
    # Create synthetic ETF data
    logger.info("Generating synthetic ETF data")
    
    # Define ETF tickers by category
    etf_dict = {
        'Treasury': ['IEF', 'TLT', 'SHY'],  # Intermediate, Long-term, Short-term Treasury
        'Corporate': ['LQD', 'VCIT', 'VCSH'],  # Investment Grade, Int. Corp, Short Corp
        'High_Yield': ['HYG', 'JNK'],  # High Yield Corporate
        'Municipal': ['MUB', 'TFI'],  # Municipal Bonds
        'International': ['BNDX', 'EMB']  # International and Emerging Markets
    }
    
    # Flatten to get all tickers
    all_tickers = [ticker for category in etf_dict.values() for ticker in category]
    
    # Create multi-level columns for ETF DataFrame
    price_data = {}
    returns_data = {}
    volatility_data = {}
    
    # Generate synthetic price series for each ETF
    for ticker in all_tickers:
        # Base properties depend on ETF category
        if ticker in etf_dict['Treasury']:
            # Correlate with treasury yields but negative (prices go up when yields go down)
            base_price = 100
            correlation = -0.7
            vol = 0.06 / np.sqrt(252)  # Lower volatility for treasuries
            beta = 0.7
        elif ticker in etf_dict['Corporate']:
            base_price = 100
            correlation = -0.6
            vol = 0.10 / np.sqrt(252)
            beta = 0.8
        elif ticker in etf_dict['High_Yield']:
            base_price = 100
            correlation = -0.3  # Less correlation with treasuries
            vol = 0.15 / np.sqrt(252)  # Higher volatility
            beta = 1.1
        else:  # Municipal and International
            base_price = 100
            correlation = -0.4
            vol = 0.12 / np.sqrt(252)
            beta = 0.9
        
        # Generate price series
        prices = np.zeros(n_days)
        prices[0] = base_price
        
        # Relate to regimes and treasury movements for realistic behavior
        for i in range(1, n_days):
            regime = regimes[i]
            params = regime_params[regime]
            
            # Relate to treasury yield changes (negative correlation)
            treasury_change = yields_df['10Y'].iloc[i] - yields_df['10Y'].iloc[i-1]
            
            # Adjust volatility based on regime
            regime_vol_multiplier = 1.0
            if regime == 2:  # Falling rates regime - bullish for bonds
                regime_vol_multiplier = 0.8
                price_boost = 0.05 * beta
            elif regime == 3:  # Stressed regime
                regime_vol_multiplier = 1.5
                price_boost = -0.10 * beta  # Price drop in stress
            elif regime == 1:  # Rising rates regime - bearish for bonds
                regime_vol_multiplier = 1.2
                price_boost = -0.08 * beta
            else:  # Normal regime
                price_boost = 0.01 * beta
            
            # Daily price change including correlation with treasuries
            price_change = (
                correlation * treasury_change * beta * 5.0 +  # Effect of treasury change
                price_boost / 100.0 +  # Regime-specific drift
                np.random.normal(0, vol * regime_vol_multiplier)  # Random noise
            )
            
            # Update price with percent change
            prices[i] = prices[i-1] * (1 + price_change)
        
        # Store in dictionary
        price_data[ticker] = prices
    
    # Create a multi-level DataFrame
    # First, create individual DataFrames
    price_df = pd.DataFrame(price_data, index=date_range)
    
    # Calculate returns
    returns_df = price_df.pct_change().fillna(0)
    
    # Calculate volatility (20-day rolling)
    volatility_df = returns_df.rolling(window=20).std().fillna(method='bfill')
    
    # Create multi-level DataFrame
    etfs_data = pd.concat([
        price_df.add_prefix('price_'),
        returns_df.add_prefix('return_'),
        volatility_df.add_prefix('vol_')
    ], axis=1)
    
    # Create dictionary with all synthetic data
    synthetic_data = {
        'yields': yields_df,
        'corporate': corporate_df,
        'etfs': etfs_data
    }
    
    logger.info(f"Generated synthetic market data with {n_days} observations and {n_regimes} regimes")
    return synthetic_data
