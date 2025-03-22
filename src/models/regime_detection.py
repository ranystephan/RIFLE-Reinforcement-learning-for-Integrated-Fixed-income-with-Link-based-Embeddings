"""
Regime Detection Module for Fixed Income RL Project

This module implements regime detection models to identify different interest rate
and credit spread regimes in the market.

Methods implemented:
1. Hidden Markov Models (HMM)
2. Gaussian Mixture Models (GMM)
3. K-means clustering
4. Spectral clustering with manifold learning

Author: ranycs & cosrv
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import joblib
import os
import matplotlib.patches as mpatches
import matplotlib.dates as mdates

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FixedIncomeRegimeDetector:
    """
    Class for detecting market regimes in fixed income data.
    """
    
    def __init__(self, n_regimes: int = 4, model_type: str = 'hmm', 
                window_size: int = 21, random_state: int = 42):
        """
        Initialize the regime detector.
        
        Args:
            n_regimes: Number of regimes to detect.
            model_type: Type of model to use ('hmm', 'gmm', 'kmeans', or 'spectral')
            window_size: Size of rolling window for feature smoothing
            random_state: Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.model_type = model_type
        self.window_size = window_size
        self.random_state = random_state
        self.model = None
        self.regime_labels = None
        self.regime_names = None
        self.feature_importance = None
        self.scaler = StandardScaler()
        self.pca = None  # For dimensionality reduction if needed
    
    def _smooth_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Smooth features using a rolling window.
        
        Args:
            features: DataFrame with features
            
        Returns:
            DataFrame with smoothed features
        """
        smoothed = features.rolling(window=self.window_size, min_periods=1).mean()
        return smoothed
    
    def _reduce_dimensions(self, features: pd.DataFrame, n_components: int = 10) -> pd.DataFrame:
        """
        Reduce dimensions of features using PCA.
        
        Args:
            features: DataFrame with features
            n_components: Number of components to keep
            
        Returns:
            DataFrame with reduced dimensions
        """
        # Handle missing values (NaN) before PCA
        logger.info(f"Feature shape before handling missing values: {features.shape}")
        missing_cols = features.columns[features.isna().any()].tolist()
        if missing_cols:
            logger.warning(f"Found {len(missing_cols)} columns with missing values before PCA")
            logger.warning(f"Missing value columns: {missing_cols[:5]}{'...' if len(missing_cols) > 5 else ''}")
            
            # First try to forward fill, then backward fill any remaining NaNs
            features_filled = features.ffill().bfill()
            
            # If any NaNs remain, replace with column means
            if features_filled.isna().any().any():
                logger.warning("Some NaN values remain after forward/backward fill. Using column means.")
                # Calculate column means, replacing NaN with 0
                col_means = features_filled.mean().fillna(0)
                # Replace remaining NaNs with column means
                for col in features_filled.columns:
                    features_filled[col] = features_filled[col].fillna(col_means[col])
        else:
            features_filled = features

        # Verify no NaNs remain
        if features_filled.isna().any().any():
            # If there are still NaNs, drop those rows as a last resort
            logger.warning("Still found NaNs after imputation. Dropping affected rows.")
            features_filled = features_filled.dropna()
        
        # Fit PCA if not already fitted
        if self.pca is None:
            n_components = min(n_components, features_filled.shape[1])
            self.pca = PCA(n_components=n_components, random_state=self.random_state)
            self.pca.fit(features_filled)
        
        # Transform features
        reduced = self.pca.transform(features_filled)
        
        # Create DataFrame with reduced features
        reduced_df = pd.DataFrame(
            reduced,
            index=features_filled.index,
            columns=[f"PC{i+1}" for i in range(reduced.shape[1])]
        )
        
        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'feature': features.columns,
            'importance': np.abs(self.pca.components_[0])
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Reduced dimensions from {features.shape[1]} to {reduced_df.shape[1]}")
        logger.info(f"Explained variance ratio: {self.pca.explained_variance_ratio_}")
        
        return reduced_df
    
    def fit(self, features: pd.DataFrame, 
            reduce_dim: bool = True, n_components: int = 10,
            smooth: bool = True) -> List[int]:
        """
        Fit the regime detection model to the data.
        
        Args:
            features: DataFrame with features
            reduce_dim: Whether to reduce dimensions with PCA
            n_components: Number of PCA components if reducing dimensions
            smooth: Whether to smooth features with a rolling window
            
        Returns:
            List of regime labels
        """
        logger.info(f"Fitting {self.model_type} model with {self.n_regimes} regimes")
        
        # Make a copy to avoid modifying the original
        X = features.copy()
        
        # Check for missing values in the input data
        missing_count = X.isna().sum().sum()
        if missing_count > 0:
            missing_pct = missing_count / (X.shape[0] * X.shape[1]) * 100
            logger.warning(f"Input data contains {missing_count} missing values ({missing_pct:.2f}% of all values)")
        
        # Smooth features if requested
        if smooth:
            X = self._smooth_features(X)
        
        # Handle any NaNs before scaling
        if X.isna().any().any():
            logger.warning("Filling missing values before scaling")
            # Forward fill and backward fill
            X = X.ffill().bfill()
            
            # If any NaNs remain, replace with column means
            if X.isna().any().any():
                col_means = X.mean().fillna(0)
                for col in X.columns:
                    X[col] = X[col].fillna(col_means[col])
        
        # Verify no NaNs remain
        if X.isna().any().any():
            logger.warning("Dropping rows with NaN values that couldn't be imputed")
            X = X.dropna()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Reduce dimensions if requested
        if reduce_dim:
            X_reduced = self._reduce_dimensions(pd.DataFrame(X_scaled, index=X.index, columns=X.columns), n_components)
            X_scaled = X_reduced.values
        
        # Fit the appropriate model
        if self.model_type == 'hmm':
            # Hidden Markov Model
            self.model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=1000,
                random_state=self.random_state
            )
            self.model.fit(X_scaled)
            
            # Predict hidden states
            self.regime_labels = self.model.predict(X_scaled)
            
        elif self.model_type == 'gmm':
            # Gaussian Mixture Model
            self.model = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type="full",
                random_state=self.random_state
            )
            self.model.fit(X_scaled)
            
            # Predict cluster assignments
            self.regime_labels = self.model.predict(X_scaled)
            
        elif self.model_type == 'kmeans':
            # K-means clustering
            self.model = KMeans(
                n_clusters=self.n_regimes,
                random_state=self.random_state
            )
            self.model.fit(X_scaled)
            
            # Predict cluster assignments
            self.regime_labels = self.model.predict(X_scaled)
            
        elif self.model_type == 'spectral':
            # Spectral clustering
            self.model = SpectralClustering(
                n_clusters=self.n_regimes,
                random_state=self.random_state,
                affinity='nearest_neighbors'
            )
            self.regime_labels = self.model.fit_predict(X_scaled)
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        logger.info(f"Model fitted successfully")
        
        # Create a DataFrame with regime labels
        self.regime_df = pd.DataFrame({
            'regime': self.regime_labels
        }, index=X.index)
        
        return self.regime_labels
    
    def predict(self, features: pd.DataFrame, 
            reduce_dim: bool = True,
            smooth: bool = True) -> List[int]:
        """
        Predict regimes for new data.
        
        Args:
            features: DataFrame with features
            reduce_dim: Whether to reduce dimensions with PCA
            smooth: Whether to smooth features with a rolling window
            
        Returns:
            List of regime labels
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        logger.info(f"Predicting regimes for new data")
        
        # Make a copy to avoid modifying the original
        X = features.copy()
        
        # Check for missing values in the input data
        missing_count = X.isna().sum().sum()
        if missing_count > 0:
            missing_pct = missing_count / (X.shape[0] * X.shape[1]) * 100
            logger.warning(f"Input data contains {missing_count} missing values ({missing_pct:.2f}% of all values)")
        
        # Smooth features if requested
        if smooth:
            X = self._smooth_features(X)
        
        # Handle any NaNs before scaling
        if X.isna().any().any():
            logger.warning("Filling missing values before prediction")
            # Forward fill and backward fill
            X = X.ffill().bfill()
            
            # If any NaNs remain, replace with column means
            if X.isna().any().any():
                col_means = X.mean().fillna(0)
                for col in X.columns:
                    X[col] = X[col].fillna(col_means[col])
        
        # Verify no NaNs remain
        if X.isna().any().any():
            logger.warning("Dropping rows with NaN values that couldn't be imputed")
            X = X.dropna()
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Reduce dimensions if requested
        if reduce_dim and self.pca is not None:
            X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
            X_reduced = self._reduce_dimensions(X_scaled_df, n_components=self.pca.n_components_)
            X_scaled = X_reduced.values
        
        # Predict with the appropriate model
        if self.model_type == 'hmm':
            regime_labels = self.model.predict(X_scaled)
        elif self.model_type in ['gmm', 'kmeans']:
            regime_labels = self.model.predict(X_scaled)
        elif self.model_type == 'spectral':
            # Spectral clustering is a transductive method and doesn't support predict
            # We need to refit the model with all data, which isn't ideal
            # In practice, we'd use a different method for online prediction
            logger.warning("Spectral clustering doesn't support predict(). Returning dummy labels.")
            regime_labels = np.zeros(X.shape[0], dtype=int)
        
        logger.info(f"Predicted regimes for {X.shape[0]} samples")
        
        return regime_labels
    
    def name_regimes(self, regime_df: pd.DataFrame, yields_df: pd.DataFrame, 
                     corporate_df: Optional[pd.DataFrame] = None) -> Dict[str, str]:
        """
        Name regimes based on yield curve and credit spread characteristics.
        
        Args:
            regime_df: DataFrame with regime labels
            yields_df: DataFrame with yield curve data
            corporate_df: DataFrame with corporate bond data
            
        Returns:
            Dictionary mapping regime labels to names
        """
        logger.info("Naming regimes")
        
        # First, validate and fix the date indices for all input dataframes
        logger.info("Validating date indices before naming regimes")
        regime_df, yields_df, corp_df = self._validate_date_indices(regime_df, yields_df, corporate_df)
        
        # Use fixed corporate_df
        corporate_df = corp_df
        
        # Create a copy of regime DataFrame
        regime_data = regime_df.copy()
        
        # Ensure we have a DatetimeIndex (should be redundant after validation, but keeping for safety)
        if not isinstance(regime_data.index, pd.DatetimeIndex):
            try:
                regime_data.index = pd.to_datetime(regime_data.index)
            except:
                logger.warning("Could not convert regime index to datetime")
        
        # Initialize a dictionary to store regime names
        regime_names = {}
        
        # Get unique regimes
        unique_regimes = regime_data['regime'].unique()
        
        # Define key features for naming
        key_features = {
            'level': None,  # Yield level
            'slope': None,  # Yield curve slope
            'curvature': None,  # Yield curve curvature
            'credit_spread': None,  # Credit spread level
            'volatility': None,  # Market volatility
        }
        
        # Calculate characteristics for each regime
        for regime in unique_regimes:
            # Get dates for this regime
            regime_dates = regime_data[regime_data['regime'] == regime].index
            
            # Extract yield data for this regime
            regime_yields = yields_df.loc[yields_df.index.isin(regime_dates)]
            
            # Calculate yield level (using 10Y if available, else average)
            if '10Y' in regime_yields.columns:
                level = regime_yields['10Y'].mean()
            else:
                # Use average of available yield columns
                yield_cols = [col for col in regime_yields.columns if col in 
                            ['2Y', '3Y', '5Y', '7Y', '10Y', '20Y', '30Y']]
                if yield_cols:
                    level = regime_yields[yield_cols].mean().mean()
                else:
                    level = np.nan
            
            # Calculate yield curve slope
            if '2Y' in regime_yields.columns and '10Y' in regime_yields.columns:
                slope = regime_yields['10Y'].mean() - regime_yields['2Y'].mean()
            elif '2s10s_spread' in regime_yields.columns:
                slope = regime_yields['2s10s_spread'].mean()
            else:
                slope = np.nan
            
            # Calculate yield curve curvature
            if all(x in regime_yields.columns for x in ['2Y', '5Y', '10Y']):
                curvature = 2 * regime_yields['5Y'].mean() - regime_yields['2Y'].mean() - regime_yields['10Y'].mean()
            elif 'curvature' in regime_yields.columns:
                curvature = regime_yields['curvature'].mean()
            else:
                curvature = np.nan
            
            # Calculate credit spread
            if corporate_df is not None:
                regime_corp = corporate_df.loc[corporate_df.index.isin(regime_dates)]
                
                if 'AAA_BAA_spread' in regime_corp.columns:
                    credit_spread = regime_corp['AAA_BAA_spread'].mean()
                elif all(x in regime_corp.columns for x in ['AAA', 'BAA']):
                    credit_spread = regime_corp['BAA'].mean() - regime_corp['AAA'].mean()
                else:
                    credit_spread = np.nan
            else:
                credit_spread = np.nan
            
            # Store regime characteristics
            key_features['level'] = level
            key_features['slope'] = slope
            key_features['curvature'] = curvature
            key_features['credit_spread'] = credit_spread
            
            # Classify yield level
            if pd.notna(level):
                if level < 1.0:
                    level_name = "Low-Yield"
                elif level < 3.0:
                    level_name = "Moderate-Yield"
                else:
                    level_name = "High-Yield"
            else:
                level_name = "Unknown-Yield"
            
            # Classify yield curve slope
            if pd.notna(slope):
                if slope < -0.1:
                    slope_name = "Inverted-Curve"
                elif slope < 0.5:
                    slope_name = "Flat-Curve"
                elif slope < 1.5:
                    slope_name = "Normal-Curve"
                else:
                    slope_name = "Steep-Curve"
            else:
                slope_name = "Unknown-Curve"
            
            # Classify credit spread
            if pd.notna(credit_spread):
                if credit_spread < 0.5:
                    spread_name = "Tight-Spread"
                elif credit_spread < 1.5:
                    spread_name = "Normal-Spread"
                else:
                    spread_name = "Wide-Spread"
            else:
                spread_name = "Unknown-Spread"
            
            # Create distinctive name for this specific regime
            # Add additional identifiers based on regime number to ensure uniqueness
            if regime == 0:
                prefix = "Growth"
            elif regime == 1:
                prefix = "Recession"
            elif regime == 2:
                prefix = "Recovery"
            elif regime == 3:
                prefix = "Crisis"
            else:
                prefix = f"Regime-{regime}"
            
            # Create full regime name
            regime_name = f"{prefix}: {level_name}/{slope_name}/{spread_name}"
            
            # Store in dictionary
            regime_key = f"Regime {regime}"
            regime_names[regime_key] = regime_name
        
        logger.info(f"Named regimes: {regime_names}")
        return regime_names
    
    def get_regime_transitions(self, regime_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate regime transition probabilities.
        
        Args:
            regime_df: DataFrame with regime labels
            
        Returns:
            DataFrame with transition probabilities
        """
        logger.info(f"Calculating regime transitions")
        
        # Get unique regimes
        regimes = sorted(regime_df['regime'].unique())
        n_regimes = len(regimes)
        
        # Create a transition count matrix
        transitions = np.zeros((n_regimes, n_regimes))
        
        # Count transitions
        prev_regime = None
        for regime in regime_df['regime']:
            if prev_regime is not None:
                transitions[prev_regime, regime] += 1
            prev_regime = regime
        
        # Convert to probabilities
        transition_probs = np.zeros((n_regimes, n_regimes))
        for i in range(n_regimes):
            row_sum = np.sum(transitions[i, :])
            if row_sum > 0:
                transition_probs[i, :] = transitions[i, :] / row_sum
        
        # Create a DataFrame with appropriate labels that include regime numbers
        if self.regime_names is not None:
            # Include numeric identifiers in index and column names
            regime_names = [f"R{r}:{self.regime_names[r]}" for r in regimes]
        else:
            regime_names = [f"Regime {r}" for r in regimes]
        
        transition_df = pd.DataFrame(
            transition_probs,
            index=regime_names,
            columns=regime_names
        )
        
        return transition_df
    
    def get_regime_statistics(self, regime_df: pd.DataFrame, yields_df: pd.DataFrame,
                        corporate_df: pd.DataFrame = None,
                        etf_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculate statistics for each regime.
        
        Args:
            regime_df: DataFrame with regime labels
            yields_df: DataFrame with yield curve data
            corporate_df: DataFrame with corporate bond data
            etf_df: DataFrame with ETF data
            
        Returns:
            DataFrame with regime statistics
        """
        logger.info(f"Calculating regime statistics")
        
        # First, validate and fix the date indices for all input dataframes
        logger.info("Validating date indices before calculating statistics")
        regime_df, yields_df, corp_df, fixed_etf_df = self._validate_date_indices(
            regime_df, yields_df, corporate_df, etf_df
        )
        
        # Use fixed dataframes
        corporate_df = corp_df
        etf_df = fixed_etf_df
        
        # Get unique regimes
        regimes = sorted(regime_df['regime'].unique())
        
        # Initialize statistics dictionary
        stats = {}
        
        # For each regime, calculate statistics
        for regime in regimes:
            # Get dates for this regime
            regime_dates = regime_df[regime_df['regime'] == regime].index
            
            # Calculate regime duration and occurrence
            duration_days = (regime_dates[-1] - regime_dates[0]).days
            occurrence_pct = len(regime_dates) / len(regime_df) * 100
            
            # Extract yield curve statistics
            regime_yields = yields_df.loc[yields_df.index.isin(regime_dates)]
            
            yield_stats = {}
            for col in ['2Y', '5Y', '10Y', '30Y', '2s10s_spread']:
                if col in regime_yields.columns:
                    yield_stats[f'avg_{col}'] = regime_yields[col].mean()
                    yield_stats[f'std_{col}'] = regime_yields[col].std()
            
            # Yield trend calculation
            if '10Y' in regime_yields.columns and len(regime_yields) > 20:
                # Use linear regression to estimate trend
                x = np.arange(len(regime_yields))
                y = regime_yields['10Y'].values
                yield_trend = np.polyfit(x, y, 1)[0] * 100  # Scale by 100 for readability
                yield_stats['yield_trend'] = yield_trend
            
            # Extract corporate bond statistics
            corp_stats = {}
            if corporate_df is not None:
                regime_corp = corporate_df.loc[corporate_df.index.isin(regime_dates)]
                
                for col in ['AAA_BAA_spread', 'High_Yield_Index']:
                    if col in regime_corp.columns:
                        corp_stats[f'avg_{col}'] = regime_corp[col].mean()
                        corp_stats[f'std_{col}'] = regime_corp[col].std()
            
            # Extract ETF statistics
            etf_stats = {}
            if etf_df is not None and not etf_df.empty:
                logger.info(f"Processing ETF data for regime {regime} with {len(etf_df)} rows")
                # Filter for just this regime's dates
                regime_etf = etf_df.loc[etf_df.index.isin(regime_dates)]
                
                if regime_etf.empty:
                    logger.warning(f"No ETF data found for regime {regime} - check date alignment")
                else:
                    logger.info(f"Found {len(regime_etf)} ETF data points for regime {regime}")
                    return_cols = [col for col in regime_etf.columns if 'return' in col and not '20d' in col and not '30d' in col]
                    
                    if len(return_cols) == 0:
                        logger.warning(f"No return columns found in ETF data for regime {regime}. Available columns: {regime_etf.columns.tolist()}")
                    else:
                        logger.info(f"Processing {len(return_cols)} ETF return columns: {return_cols}")
                        
                        for col in return_cols:
                            # Verify we have valid data before calculating statistics
                            if not regime_etf[col].isna().all():
                                etf_stats[f'avg_{col}'] = regime_etf[col].mean() * 252  # Annualize
                                etf_stats[f'std_{col}'] = regime_etf[col].std() * np.sqrt(252)  # Annualize
                            else:
                                logger.warning(f"Column {col} contains all NaN values for regime {regime}")
            else:
                logger.warning(f"No ETF data available for regime statistics calculation")
            
            # Generate a summary description
            summary = []
            
            # Yield level
            avg_10y = yield_stats.get('avg_10Y', 0)
            if avg_10y < 1.5:
                summary.append("Very Low Yields")
            elif avg_10y < 2.0:
                summary.append("Low Yields")
            elif avg_10y < 2.5:
                summary.append("Low-Moderate Yields")
            elif avg_10y < 3.0:
                summary.append("Moderate Yields")
            elif avg_10y < 3.5:
                summary.append("Moderate-High Yields")
            else:
                summary.append("High Yields")
                
            # Yield curve
            avg_2s10s = yield_stats.get('avg_2s10s_spread', 0)
            if 'avg_2s10s_spread' in yield_stats:
                if avg_2s10s < 0:
                    summary.append("Inverted Curve")
                elif avg_2s10s < 0.5:
                    summary.append("Flat Curve")
                elif avg_2s10s > 0.8:
                    summary.append("Steep Curve")
                else:
                    summary.append("Normal Curve")
            
            # Credit spreads
            avg_aaa_baa = corp_stats.get('avg_AAA_BAA_spread', 0)
            if avg_aaa_baa > 1.5:
                summary.append("Wide Credit Spreads")
            elif avg_aaa_baa > 0.7:
                summary.append("Moderate Credit Spreads")
            elif avg_aaa_baa > 0.65:
                summary.append("Tight Credit Spreads")
            elif avg_aaa_baa < 0.65:
                summary.append("Very Tight Credit Spreads")
                
            # Yield trend
            yield_trend_val = yield_stats.get('yield_trend', 0)
            if yield_trend_val > 1.0:
                summary.append("Rapidly Rising Rates")
            elif yield_trend_val > 0.5:
                summary.append("Rising Rates")
            elif yield_trend_val < -1.0:
                summary.append("Rapidly Falling Rates")
            elif yield_trend_val < -0.5:
                summary.append("Falling Rates")
                
            # Combine all statistics
            if self.regime_names is not None:
                # Include numerical regime identifier in the name
                regime_name = f"R{regime}:{self.regime_names[regime]}"
            else:
                regime_name = f"Regime {regime}"
            
            stats[regime_name] = {
                'duration_days': duration_days,
                'occurrence_pct': occurrence_pct,
                'summary': ", ".join(summary),
                **yield_stats,
                **corp_stats,
                **etf_stats
            }
        
        # Convert to DataFrame
        stats_df = pd.DataFrame(stats).T
        
        # Ensure the summary is displayed first
        if 'summary' in stats_df.columns:
            cols = ['duration_days', 'occurrence_pct', 'summary'] + [col for col in stats_df.columns if col not in ['duration_days', 'occurrence_pct', 'summary']]
            stats_df = stats_df[cols]
        
        return stats_df
    
    def plot_regimes(self, regime_df: pd.DataFrame, yields_df: pd.DataFrame,
                corporate_df: pd.DataFrame = None,
                figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot the detected regimes along with key market indicators.
        
        Args:
            regime_df: DataFrame with regime labels
            yields_df: DataFrame with yield curve data
            corporate_df: DataFrame with corporate bond data
            figsize: Figure size as (width, height)
            
        Returns:
            Matplotlib figure
        """
        logger.info(f"Plotting regimes")
        
        # First, validate and fix the date indices for all input dataframes
        logger.info("Validating date indices before plotting")
        regime_df, yields_df, corp_df = self._validate_date_indices(regime_df, yields_df, corporate_df)
        
        # Use fixed corporate_df
        corporate_df = corp_df
        
        # Additional check: Ensure dates start and end according to expected ranges
        for df_name, df in [("regime_df", regime_df), ("yields_df", yields_df), 
                           ("corporate_df", corporate_df if corporate_df is not None else None)]:
            if df is not None and isinstance(df.index, pd.DatetimeIndex):
                # Look for Unix epoch dates (1970)
                if df.index.min().year < 1980:
                    logger.warning(f"{df_name} still has suspicious early dates: {df.index.min()}")
                    # Force the start date to 2010 if we need to
                    if df.index.min().year == 1970:
                        logger.info(f"Forcing {df_name} to start from 2010-01-01")
                        # Create a new date range
                        new_index = pd.date_range(start='2010-01-01', periods=len(df), freq='B')
                        df.index = new_index
        
        # Ensure all dataframes have proper DatetimeIndex after our validation
        for df_name, df in [("regime_df", regime_df), ("yields_df", yields_df), 
                           ("corporate_df", corporate_df if corporate_df is not None else None)]:
            if df is not None and not isinstance(df.index, pd.DatetimeIndex):
                logger.warning(f"{df_name} index is not a DatetimeIndex. Attempting to convert again.")
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception as e:
                    logger.error(f"Failed to convert {df_name} index to DatetimeIndex: {str(e)}")
        
        # Get the common date range across all dataframes to ensure consistency
        min_date = max(df.index.min() for df in [regime_df, yields_df, corporate_df] if df is not None)
        max_date = min(df.index.max() for df in [regime_df, yields_df, corporate_df] if df is not None)
        
        # Log the date ranges for debugging
        logger.info(f"Regime data date range: {regime_df.index.min()} to {regime_df.index.max()}")
        logger.info(f"Yields data date range: {yields_df.index.min()} to {yields_df.index.max()}")
        if corporate_df is not None:
            logger.info(f"Corporate data date range: {corporate_df.index.min()} to {corporate_df.index.max()}")
        logger.info(f"Using common date range for plotting: {min_date} to {max_date}")
        
        # Create figure and axis
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        # Get unique regimes
        regimes = sorted(regime_df['regime'].unique())
        
        # Use regime names if available
        if self.regime_names is not None:
            regime_names = {r: f"R{r}: {self.regime_names[r]}" for r in regimes}
        else:
            regime_names = {r: f"Regime {r}" for r in regimes}
        
        # Filter data to common date range
        regime_df_plot = regime_df.loc[min_date:max_date]
        yields_df_plot = yields_df.loc[min_date:max_date]
        corporate_df_plot = corporate_df.loc[min_date:max_date] if corporate_df is not None else None
        
        # Plot 10Y yield
        if '10Y' in yields_df_plot.columns:
            axes[0].plot(yields_df_plot.index, yields_df_plot['10Y'], 'k-', label='10Y Yield')
            axes[0].set_ylabel('Yield (%)')
            axes[0].legend()
            axes[0].grid(True)
        
        # Plot 2s10s spread
        if '2s10s_spread' in yields_df_plot.columns:
            axes[1].plot(yields_df_plot.index, yields_df_plot['2s10s_spread'], 'b-', label='2s10s Spread')
            axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[1].set_ylabel('Spread (%)')
            axes[1].legend()
            axes[1].grid(True)
        
        # Plot AAA-BAA spread
        if corporate_df_plot is not None and 'AAA_BAA_spread' in corporate_df_plot.columns:
            axes[2].plot(corporate_df_plot.index, corporate_df_plot['AAA_BAA_spread'], 'g-', label='AAA-BAA Spread')
            axes[2].set_ylabel('Spread (%)')
            axes[2].legend()
            axes[2].grid(True)
        
        # Add regime background colors to all subplots
        colors = plt.cm.tab10(np.linspace(0, 1, len(regimes)))
        regime_colors = {r: colors[i] for i, r in enumerate(regimes)}
        
        # Previous regime for change detection
        prev_regime = None
        regime_start = None
        
        # Add vertical lines at regime changes and colored backgrounds
        for date, row in regime_df_plot.iterrows():
            if prev_regime is None or prev_regime != row['regime']:
                # Regime change detected
                if prev_regime is not None:
                    # Add vertical line
                    for ax in axes:
                        ax.axvline(x=date, color='gray', linestyle='--', alpha=0.5)
                    
                    # Add background color for previous regime
                    for ax in axes:
                        ax.axvspan(regime_start, date, alpha=0.2, color=regime_colors[prev_regime])
                
                # Store new regime info
                prev_regime = row['regime']
                regime_start = date
        
        # Add background for last regime
        if prev_regime is not None:
            for ax in axes:
                ax.axvspan(regime_start, regime_df_plot.index[-1], alpha=0.2, color=regime_colors[prev_regime])
        
        # Add regime legend
        patches = [mpatches.Patch(color=regime_colors[r], alpha=0.5, label=regime_names[r]) for r in regimes]
        axes[0].legend(handles=patches, loc='upper right')
        
        # Set title and labels
        fig.suptitle('Fixed Income Market Regimes', fontsize=16)
        axes[-1].set_xlabel('Date')
        
        # Format x-axis as dates with improved formatting
        # Determine appropriate locator based on date range
        date_range_years = (max_date - min_date).days / 365.25
        
        for ax in axes:
            # Set the x-axis limits explicitly to our date range
            ax.set_xlim(min_date, max_date)
            
            # Choose appropriate date locators based on the range
            if date_range_years <= 2:
                # For shorter ranges (<=2 years), show months
                ax.xaxis.set_major_locator(mdates.MonthLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                if date_range_years <= 0.5:  # Less than 6 months
                    ax.xaxis.set_minor_locator(mdates.WeekdayLocator())
            elif date_range_years <= 5:
                # For medium ranges (<=5 years), show quarters
                ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                ax.xaxis.set_minor_locator(mdates.MonthLocator())
            else:
                # For longer ranges (>5 years), show years with month minor ticks
                ax.xaxis.set_major_locator(mdates.YearLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 7)))  # Jan and Jul
        
        # Rotate date labels for better readability
        fig.autofmt_xdate(rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        return fig
    
    def plot_regime_characteristics(self, stats_df: pd.DataFrame, 
                                figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot characteristics of each regime.
        
        Args:
            stats_df: DataFrame with regime statistics
            figsize: Figure size as (width, height)
            
        Returns:
            Matplotlib figure
        """
        logger.info(f"Plotting regime characteristics")
        
        # Create figure and axis
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot yield curves for each regime
        ax = axes[0, 0]
        tenors = []
        for col in stats_df.columns:
            if col.startswith('avg_') and col.endswith('Y'):
                tenor = col.replace('avg_', '')
                tenors.append(tenor)
        
        if tenors:
            tenors = sorted(tenors, key=lambda x: float(x.replace('Y', '')))
            tenor_values = [float(t.replace('Y', '')) for t in tenors]
            
            for regime in stats_df.index:
                yields = [stats_df.loc[regime, f'avg_{t}'] for t in tenors]
                ax.plot(tenor_values, yields, 'o-', label=regime)
            
            ax.set_xlabel('Tenor (years)')
            ax.set_ylabel('Yield (%)')
            ax.set_title('Average Yield Curve by Regime')
            ax.grid(True)
            
            # Add legend with better formatting
            if len(stats_df.index) <= 6:
                # If few regimes, keep legend in the chart
                ax.legend(loc='best')
            else:
                # If many regimes, move legend outside
                ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        else:
            ax.text(0.5, 0.5, 'No yield curve data available', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
        
        # Plot spread statistics
        ax = axes[0, 1]
        spread_cols = [col for col in stats_df.columns if 'spread' in col and 'avg' in col]
        
        if len(spread_cols) > 0:
            # Create a custom DataFrame for better visualization
            spread_df = stats_df[spread_cols].copy()
            # Rename columns for better display
            spread_df.columns = [col.replace('avg_', '').replace('_spread', '') for col in spread_cols]
            spread_df.plot(kind='bar', ax=ax, rot=45)
            ax.set_ylabel('Spread (%)')
            ax.set_title('Average Spreads by Regime')
            
            # Only show legend if there are multiple columns
            if len(spread_cols) > 1:
                ax.legend(loc='best', fontsize='small')
            else:
                ax.legend().set_visible(False)
                
            # Adjust to show all x labels
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, 'No spread data available', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
        
        # Plot return statistics
        ax = axes[1, 0]
        return_cols = [col for col in stats_df.columns if 'return' in col and 'avg' in col]
        
        # Check if we have actual return data
        if len(return_cols) == 0:
            logger.warning("No ETF return data found in regime statistics. This may indicate that ETF data wasn't properly processed.")
            logger.info("Look at the 'etf_df' parameter passed to get_regime_statistics to ensure ETF data is available.")
            
            # Instead of immediately generating synthetic data, let's inform the user
            ax.text(0.5, 0.5, 'No ETF return data available\nCheck that ETF data is being properly loaded',
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, color='red', fontsize=10)
            ax.set_title('ETF Returns by Regime (Missing Data)')
        else:
            # Create a custom DataFrame for better visualization
            return_df = stats_df[return_cols].copy()
            # Rename columns for better display
            return_df.columns = [col.replace('avg_', '').replace('_return', '') for col in return_cols]
            return_df.plot(kind='bar', ax=ax, rot=45)
            ax.set_ylabel('Annualized Return (%)')
            ax.set_title('Average Returns by Regime')
            
            # Only show legend if there are multiple columns
            if len(return_cols) > 1:
                ax.legend(loc='best', fontsize='small')
            else:
                ax.legend().set_visible(False)
                
            # Adjust to show all x labels
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot regime occurrence
        ax = axes[1, 1]
        if not stats_df.empty and 'occurrence_pct' in stats_df.columns:
            # Make sure regime labels are clear and properly formatted
            labels = [f"{idx.split(':')[0] if ':' in idx else idx} ({stats_df.loc[idx, 'occurrence_pct']:.1f}%)" 
                    for idx in stats_df.index]
            wedges, texts, autotexts = ax.pie(
                stats_df['occurrence_pct'].values, 
                labels=labels if len(labels) <= 5 else None,  # Only show labels directly if 5 or fewer regimes
                autopct='%1.1f%%', 
                startangle=90,
                wedgeprops=dict(width=0.5, edgecolor='w')  # Make it a donut chart
            )
            
            # If more than 5 regimes, create a legend instead of direct labels
            if len(labels) > 5:
                ax.legend(wedges, labels, title="Regimes", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
                
            # Make the percentage labels more readable
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                
            ax.set_title('Regime Occurrence')
        else:
            ax.text(0.5, 0.5, 'No regime occurrence data available', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
        
        # Set layout with extra space for legends
        plt.tight_layout()
        
        return fig
    
    def save_model(self, filepath: str):
        """
        Save the fitted model and associated data.
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'n_regimes': self.n_regimes,
            'window_size': self.window_size,
            'scaler': self.scaler,
            'pca': self.pca,
            'regime_names': self.regime_names,
            'feature_importance': self.feature_importance
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'FixedIncomeRegimeDetector':
        """
        Load a saved model.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded FixedIncomeRegimeDetector instance
        """
        model_data = joblib.load(filepath)
        
        # Create a new instance
        detector = cls(
            n_regimes=model_data['n_regimes'],
            model_type=model_data['model_type'],
            window_size=model_data['window_size']
        )
        
        # Load model components
        detector.model = model_data['model']
        detector.scaler = model_data['scaler']
        detector.pca = model_data['pca']
        detector.regime_names = model_data['regime_names']
        detector.feature_importance = model_data['feature_importance']
        
        logger.info(f"Model loaded from {filepath}")
        return detector

    def _validate_date_indices(self, *dataframes):
        """
        Validate and fix date indices in the provided DataFrames.
        
        This method checks for common datetime index issues and fixes them:
        1. Indices starting at 1970 (Unix epoch) which indicates timestamp conversion issues
        2. Indices with nanosecond precision
        3. RangeIndex mistakenly used instead of DatetimeIndex
        
        Args:
            *dataframes: DataFrames to validate/fix (regime_df, yields_df, corporate_df, etf_df in order)
            
        Returns:
            List of DataFrames with fixed indices
        """
        fixed_dfs = []
        first_valid_dates = []
        last_valid_dates = []
        
        # Define dataframe names for better logging
        df_names = ['regime_df', 'yields_df', 'corporate_df', 'etf_df']
        
        # First pass: identify valid date ranges across datasets
        for i, df in enumerate(dataframes):
            df_name = df_names[i] if i < len(df_names) else f"DataFrame {i}"
            
            if df is None:
                logger.info(f"{df_name} is None, skipping validation")
                fixed_dfs.append(None)
                continue
                
            logger.info(f"Validating {df_name} with {len(df)} rows")
            df_copy = df.copy()
            
            # Convert to DatetimeIndex if not already
            if not isinstance(df_copy.index, pd.DatetimeIndex):
                try:
                    logger.info(f"{df_name} index is not DatetimeIndex, converting...")
                    df_copy.index = pd.to_datetime(df_copy.index)
                except Exception as e:
                    logger.warning(f"{df_name}: Failed to convert index to DatetimeIndex: {str(e)}")
                    
            # Check for 1970 dates (Unix epoch) which indicate timestamp conversion issues
            suspicious_dates = (df_copy.index.year == 1970).sum() if isinstance(df_copy.index, pd.DatetimeIndex) else 0
            if suspicious_dates > 0:
                logger.warning(f"{df_name}: Found {suspicious_dates} suspicious dates from 1970 (Unix epoch)")
                
                # Create a new proper date range - default to 2010-2024 business days
                try:
                    # Create a business day date range with the right number of periods
                    corrected_index = pd.date_range(
                        start="2010-01-01", 
                        periods=len(df_copy),
                        freq='B'
                    )
                    
                    df_copy.index = corrected_index
                    logger.info(f"{df_name}: Reconstructed date index from {corrected_index[0]} to {corrected_index[-1]}")
                except Exception as e:
                    logger.error(f"Failed to reconstruct date index for {df_name}: {str(e)}")
            
            # Store valid date range info
            if isinstance(df_copy.index, pd.DatetimeIndex) and len(df_copy) > 0:
                if df_copy.index.min().year > 1980:  # Avoid clearly wrong dates
                    first_valid_dates.append(df_copy.index.min())
                    logger.info(f"{df_name} has valid start date: {df_copy.index.min()}")
                if df_copy.index.max().year < 2100:  # Avoid clearly wrong dates
                    last_valid_dates.append(df_copy.index.max())
                    logger.info(f"{df_name} has valid end date: {df_copy.index.max()}")
            
            # For ETF data specifically, log column information
            if i == 3 and df_name == 'etf_df':
                if df_copy is not None and not df_copy.empty:
                    return_cols = [col for col in df_copy.columns if 'return' in col]
                    logger.info(f"ETF data contains {len(return_cols)} return columns: {return_cols}")
                    
                    # Check for missing values
                    missing_values = df_copy[return_cols].isna().sum().sum() if return_cols else 0
                    if missing_values > 0:
                        logger.warning(f"ETF data contains {missing_values} missing values in return columns")
            
            fixed_dfs.append(df_copy)
                
        # Determine consensus date range from all valid data
        if first_valid_dates and last_valid_dates:
            consensus_start = min(first_valid_dates)
            consensus_end = max(last_valid_dates)
            logger.info(f"Consensus date range across datasets: {consensus_start} to {consensus_end}")
            
            # Second pass: Ensure all DataFrames cover the same date range if required
            for i, df in enumerate(fixed_dfs):
                df_name = df_names[i] if i < len(df_names) else f"DataFrame {i}"
                
                if df is None:
                    continue
                
                # If index is still suspicious after first pass, try to align with other valid dataframes
                if isinstance(df.index, pd.DatetimeIndex):
                    is_suspicious = df.index.min().year <= 1980 or df.index.max().year >= 2100
                    
                    if is_suspicious:
                        logger.warning(f"{df_name}: Still has suspicious dates, will align with consensus range")
                        
                        try:
                            # Create a business day date range aligned with the consensus
                            corrected_index = pd.date_range(
                                start=consensus_start, 
                                end=consensus_end,
                                freq='B'
                            )
                            
                            # If lengths don't match, choose a subset or extend as needed
                            if len(corrected_index) > len(df):
                                logger.warning(f"{df_name}: Consensus range longer than dataframe, will use subset")
                                corrected_index = corrected_index[:len(df)]
                            elif len(corrected_index) < len(df):
                                logger.warning(f"{df_name}: Dataframe longer than consensus range, will extend range")
                                # Extend backwards from start date
                                extension = pd.date_range(
                                    end=consensus_start - pd.Timedelta(days=1),
                                    periods=len(df) - len(corrected_index),
                                    freq='B'
                                )
                                corrected_index = extension.append(corrected_index)
                                
                            fixed_dfs[i].index = corrected_index
                            logger.info(f"{df_name}: Aligned to date range {corrected_index[0]} to {corrected_index[-1]}")
                        except Exception as e:
                            logger.error(f"Failed to align {df_name} with consensus range: {str(e)}")
        
        return fixed_dfs
