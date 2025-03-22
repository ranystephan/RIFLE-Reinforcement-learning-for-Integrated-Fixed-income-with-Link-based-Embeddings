"""
Performance Metrics Module for Fixed Income RL Project

This module implements performance metrics for evaluating fixed income strategies:
1. Return metrics (annualized return, Sharpe ratio, etc.)
2. Risk metrics (volatility, drawdown, VaR, etc.)
3. Duration-related metrics (duration error, convexity, etc.)
4. Regime-dependent metrics (alpha, beta, etc.)

Mathematical foundations:
- Risk-adjusted return metrics
- Statistical performance attribution
- Fixed income-specific metrics
- Portfolio-level statistics

Author: ranycs & cosrv
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """
    Class for calculating performance metrics.
    """
    
    def __init__(self, annualization_factor: float = 252):
        """
        Initialize performance metrics calculator.
        
        Args:
            annualization_factor: Factor for annualizing returns (252 for daily, 12 for monthly, etc.)
        """
        self.annualization_factor = annualization_factor
    
    def calculate_return_metrics(self, returns: pd.Series, 
                              risk_free_rate: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate return metrics.
        
        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate (same frequency as returns)
            
        Returns:
            Dictionary with return metrics
        """
        # Calculate total return
        total_return = (1 + returns).prod() - 1
        
        # Calculate annualized return
        num_periods = len(returns)
        years = num_periods / self.annualization_factor
        annualized_return = (1 + total_return) ** (1 / years) - 1
        
        # Calculate return statistics
        mean_return = returns.mean()
        median_return = returns.median()
        std_return = returns.std()
        
        # Calculate annualized volatility
        annualized_volatility = std_return * np.sqrt(self.annualization_factor)
        
        # Calculate Sharpe ratio
        if risk_free_rate is not None:
            excess_returns = returns - risk_free_rate
            sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(self.annualization_factor)
        else:
            sharpe_ratio = np.nan
        
        # Calculate Sortino ratio
        if risk_free_rate is not None:
            downside_returns = returns[returns < risk_free_rate] - risk_free_rate
            sortino_ratio = excess_returns.mean() / downside_returns.std() * np.sqrt(self.annualization_factor) if len(downside_returns) > 0 else np.nan
        else:
            sortino_ratio = np.nan
        
        # Calculate skewness and kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Calculate value at risk
        var_95 = -np.percentile(returns, 5)
        var_99 = -np.percentile(returns, 1)
        
        # Calculate conditional value at risk (Expected Shortfall)
        cvar_95 = -returns[returns <= -var_95].mean()
        cvar_99 = -returns[returns <= -var_99].mean()
        
        # Calculate best and worst returns
        best_return = returns.max()
        worst_return = returns.min()
        
        # Calculate percentage of positive returns
        positive_returns_pct = (returns > 0).mean()
        
        # Create metrics dictionary
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'mean_return': mean_return,
            'median_return': median_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'best_return': best_return,
            'worst_return': worst_return,
            'positive_returns_pct': positive_returns_pct
        }
        
        return metrics
    
    def calculate_drawdown_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate drawdown metrics.
        
        Args:
            returns: Series of returns
            
        Returns:
            Dictionary with drawdown metrics
        """
        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative_returns.cummax()
        
        # Calculate drawdowns
        drawdowns = 1 - cumulative_returns / running_max
        
        # Calculate maximum drawdown
        max_drawdown = drawdowns.max()
        
        # Calculate average drawdown
        avg_drawdown = drawdowns.mean()
        
        # Calculate drawdown duration
        is_drawdown = drawdowns > 0
        is_start = is_drawdown & ~is_drawdown.shift(1).fillna(False)
        is_end = ~is_drawdown & is_drawdown.shift(1).fillna(False)
        
        # Get start and end indices
        start_indices = np.where(is_start)[0]
        end_indices = np.where(is_end)[0]
        
        # Adjust if needed
        if len(end_indices) < len(start_indices):
            end_indices = np.append(end_indices, len(drawdowns) - 1)
        
        # Calculate durations
        if len(start_indices) > 0 and len(end_indices) > 0:
            durations = end_indices - start_indices
            max_duration = durations.max()
            avg_duration = durations.mean()
        else:
            max_duration = 0
            avg_duration = 0
        
        # Calculate time to recovery
        if max_drawdown > 0:
            max_drawdown_idx = drawdowns.argmax()
            recovery_idx = drawdowns.iloc[max_drawdown_idx:].idxmin()
            time_to_recovery = (recovery_idx - drawdowns.index[max_drawdown_idx]).days if isinstance(recovery_idx, pd.Timestamp) else np.nan
        else:
            time_to_recovery = 0
        
        # Calculate Calmar ratio
        if max_drawdown > 0:
            # Annualized return
            total_return = (1 + returns).prod() - 1
            num_periods = len(returns)
            years = num_periods / self.annualization_factor
            annualized_return = (1 + total_return) ** (1 / years) - 1
            
            calmar_ratio = annualized_return / max_drawdown
        else:
            calmar_ratio = np.nan
        
        # Create metrics dictionary
        metrics = {
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'max_drawdown_duration': max_duration,
            'avg_drawdown_duration': avg_duration,
            'time_to_recovery': time_to_recovery,
            'calmar_ratio': calmar_ratio
        }
        
        return metrics
    
    def calculate_relative_metrics(self, returns: pd.Series, 
                               benchmark_returns: pd.Series) -> Dict[str, float]:
        """
        Calculate relative performance metrics.
        
        Args:
            returns: Series of returns
            benchmark_returns: Series of benchmark returns
            
        Returns:
            Dictionary with relative metrics
        """
        # Ensure same index
        common_idx = returns.index.intersection(benchmark_returns.index)
        returns = returns.loc[common_idx]
        benchmark_returns = benchmark_returns.loc[common_idx]
        
        # Calculate active returns
        active_returns = returns - benchmark_returns
        
        # Calculate tracking error
        tracking_error = active_returns.std() * np.sqrt(self.annualization_factor)
        
        # Calculate information ratio
        information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(self.annualization_factor) if active_returns.std() > 0 else np.nan
        
        # Calculate beta
        cov = np.cov(returns, benchmark_returns)[0, 1]
        var = np.var(benchmark_returns)
        beta = cov / var if var > 0 else np.nan
        
        # Calculate alpha
        alpha = returns.mean() * self.annualization_factor - beta * benchmark_returns.mean() * self.annualization_factor
        
        # Calculate R-squared
        if beta != np.nan and var > 0:
            corr = np.corrcoef(returns, benchmark_returns)[0, 1]
            r_squared = corr**2
        else:
            r_squared = np.nan
        
        # Calculate up/down capture
        up_market = benchmark_returns > 0
        down_market = benchmark_returns < 0
        
        if up_market.sum() > 0:
            up_capture = (returns[up_market].mean() / benchmark_returns[up_market].mean() 
                         if benchmark_returns[up_market].mean() != 0 else np.nan)
        else:
            up_capture = np.nan
        
        if down_market.sum() > 0:
            down_capture = (returns[down_market].mean() / benchmark_returns[down_market].mean()
                           if benchmark_returns[down_market].mean() != 0 else np.nan)
        else:
            down_capture = np.nan
        
        # Calculate batting average (% of periods outperforming)
        batting_average = (active_returns > 0).mean()
        
        # Calculate win/loss ratio
        wins = active_returns[active_returns > 0]
        losses = active_returns[active_returns < 0]
        
        if len(losses) > 0 and losses.mean() != 0:
            win_loss_ratio = abs(wins.mean() / losses.mean())
        else:
            win_loss_ratio = np.nan
        
        # Create metrics dictionary
        metrics = {
            'alpha': alpha,
            'beta': beta,
            'r_squared': r_squared,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'up_capture': up_capture,
            'down_capture': down_capture,
            'batting_average': batting_average,
            'win_loss_ratio': win_loss_ratio
        }
        
        return metrics
    
    def calculate_fixed_income_metrics(self, returns: pd.Series, 
                                     durations: pd.Series,
                                     target_durations: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Calculate fixed income specific metrics.
        
        Args:
            returns: Series of returns
            durations: Series of portfolio durations
            target_durations: Series of target durations (if any)
            
        Returns:
            Dictionary with fixed income metrics
        """
        # Calculate duration statistics
        avg_duration = durations.mean()
        min_duration = durations.min()
        max_duration = durations.max()
        std_duration = durations.std()
        
        # Calculate duration-adjusted return
        duration_adjusted_returns = returns / durations
        duration_adjusted_return = duration_adjusted_returns.mean() * self.annualization_factor
        
        # Calculate duration error if target durations provided
        if target_durations is not None:
            # Ensure same index
            common_idx = durations.index.intersection(target_durations.index)
            durations = durations.loc[common_idx]
            target_durations = target_durations.loc[common_idx]
            
            # Calculate duration error
            duration_error = durations - target_durations
            avg_duration_error = duration_error.mean()
            abs_duration_error = duration_error.abs()
            avg_abs_duration_error = abs_duration_error.mean()
            max_duration_error = abs_duration_error.max()
            
            # Calculate duration error statistics
            duration_error_metrics = {
                'avg_duration_error': avg_duration_error,
                'avg_abs_duration_error': avg_abs_duration_error,
                'max_duration_error': max_duration_error
            }
        else:
            duration_error_metrics = {}
        
        # Create metrics dictionary
        metrics = {
            'avg_duration': avg_duration,
            'min_duration': min_duration,
            'max_duration': max_duration,
            'std_duration': std_duration,
            'duration_adjusted_return': duration_adjusted_return,
            **duration_error_metrics
        }
        
        return metrics
    
    def calculate_portfolio_turnover(self, weights: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate portfolio turnover metrics.
        
        Args:
            weights: DataFrame with portfolio weights over time
            
        Returns:
            Dictionary with turnover metrics
        """
        # Calculate turnover for each period
        turnover = weights.diff().abs().sum(axis=1).dropna()
        
        # Calculate turnover statistics
        avg_turnover = turnover.mean()
        min_turnover = turnover.min()
        max_turnover = turnover.max()
        std_turnover = turnover.std()
        
        # Calculate annualized turnover
        annualized_turnover = avg_turnover * self.annualization_factor / 2  # Divide by 2 for one-way turnover
        
        # Create metrics dictionary
        metrics = {
            'avg_turnover': avg_turnover,
            'min_turnover': min_turnover,
            'max_turnover': max_turnover,
            'std_turnover': std_turnover,
            'annualized_turnover': annualized_turnover
        }
        
        return metrics
    
    def calculate_regime_metrics(self, returns: pd.Series,
                               benchmark_returns: pd.Series,
                               regimes: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Calculate performance metrics by regime.
        
        Args:
            returns: Series of returns
            benchmark_returns: Series of benchmark returns
            regimes: Series of regime labels
            
        Returns:
            Dictionary with metrics by regime
        """
        # Ensure same index
        common_idx = returns.index.intersection(benchmark_returns.index).intersection(regimes.index)
        returns = returns.loc[common_idx]
        benchmark_returns = benchmark_returns.loc[common_idx]
        regimes = regimes.loc[common_idx]
        
        # Get unique regimes
        unique_regimes = regimes.unique()
        
        # Calculate metrics for each regime
        regime_metrics = {}
        
        for regime in unique_regimes:
            # Filter by regime
            regime_filter = regimes == regime
            regime_returns = returns[regime_filter]
            regime_benchmark_returns = benchmark_returns[regime_filter]
            
            # Calculate metrics if there are enough data points
            if len(regime_returns) > 10:
                # Return metrics
                return_metrics = self.calculate_return_metrics(regime_returns)
                
                # Relative metrics
                relative_metrics = self.calculate_relative_metrics(regime_returns, regime_benchmark_returns)
                
                # Drawdown metrics
                drawdown_metrics = self.calculate_drawdown_metrics(regime_returns)
                
                # Combine metrics
                regime_metrics[str(regime)] = {
                    **return_metrics,
                    **relative_metrics,
                    **drawdown_metrics,
                    'num_periods': len(regime_returns)
                }
        
        return regime_metrics
    
    def calculate_all_metrics(self, returns: pd.Series,
                            benchmark_returns: Optional[pd.Series] = None,
                            risk_free_rate: Optional[float] = None,
                            durations: Optional[pd.Series] = None,
                            target_durations: Optional[pd.Series] = None,
                            weights: Optional[pd.DataFrame] = None,
                            regimes: Optional[pd.Series] = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate all performance metrics.
        
        Args:
            returns: Series of returns
            benchmark_returns: Series of benchmark returns
            risk_free_rate: Risk-free rate
            durations: Series of portfolio durations
            target_durations: Series of target durations
            weights: DataFrame with portfolio weights over time
            regimes: Series of regime labels
            
        Returns:
            Dictionary with all metrics
        """
        # Initialize metrics dictionary
        metrics = {}
        
        # Calculate return metrics
        metrics['return'] = self.calculate_return_metrics(returns, risk_free_rate)
        
        # Calculate drawdown metrics
        metrics['drawdown'] = self.calculate_drawdown_metrics(returns)
        
        # Calculate relative metrics if benchmark provided
        if benchmark_returns is not None:
            metrics['relative'] = self.calculate_relative_metrics(returns, benchmark_returns)
        
        # Calculate fixed income metrics if durations provided
        if durations is not None:
            metrics['fixed_income'] = self.calculate_fixed_income_metrics(returns, durations, target_durations)
        
        # Calculate turnover metrics if weights provided
        if weights is not None:
            metrics['turnover'] = self.calculate_portfolio_turnover(weights)
        
        # Calculate regime metrics if regimes and benchmark provided
        if regimes is not None and benchmark_returns is not None:
            metrics['regime'] = self.calculate_regime_metrics(returns, benchmark_returns, regimes)
        
        return metrics
    
    def plot_returns_histogram(self, returns: pd.Series, 
                             benchmark_returns: Optional[pd.Series] = None,
                             figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot histogram of returns.
        
        Args:
            returns: Series of returns
            benchmark_returns: Series of benchmark returns
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot strategy returns
        sns.histplot(returns, bins=30, kde=True, ax=ax, label='Strategy')
        
        # Plot benchmark returns if provided
        if benchmark_returns is not None:
            sns.histplot(benchmark_returns, bins=30, kde=True, ax=ax, alpha=0.7, label='Benchmark')
        
        # Add labels and title
        ax.set_xlabel('Return')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Returns')
        ax.legend()
        
        # Add vertical line at mean
        ax.axvline(returns.mean(), color='blue', linestyle='--', alpha=0.8)
        if benchmark_returns is not None:
            ax.axvline(benchmark_returns.mean(), color='orange', linestyle='--', alpha=0.8)
        
        return fig
    
    def plot_drawdowns(self, returns: pd.Series, 
                      figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot drawdowns over time.
        
        Args:
            returns: Series of returns
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative_returns.cummax()
        
        # Calculate drawdowns
        drawdowns = 1 - cumulative_returns / running_max
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot drawdowns
        drawdowns.plot(ax=ax)
        
        # Add labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown')
        ax.set_title('Portfolio Drawdowns')
        ax.grid(True)
        
        # Invert y-axis for better visualization
        ax.invert_yaxis()
        
        return fig
    
    def plot_regime_returns(self, returns: pd.Series, regimes: pd.Series,
                          figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot returns by regime.
        
        Args:
            returns: Series of returns
            regimes: Series of regime labels
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Ensure same index
        common_idx = returns.index.intersection(regimes.index)
        returns = returns.loc[common_idx]
        regimes = regimes.loc[common_idx]
        
        # Get unique regimes
        unique_regimes = regimes.unique()
        
        # Calculate average return by regime
        regime_returns = {}
        regime_volatilities = {}
        
        for regime in unique_regimes:
            regime_filter = regimes == regime
            regime_returns[str(regime)] = returns[regime_filter].mean() * self.annualization_factor
            regime_volatilities[str(regime)] = returns[regime_filter].std() * np.sqrt(self.annualization_factor)
        
        # Create DataFrame
        df = pd.DataFrame({
            'annualized_return': regime_returns,
            'annualized_volatility': regime_volatilities
        })
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot as bar chart
        df['annualized_return'].plot(kind='bar', ax=ax)
        
        # Add labels and title
        ax.set_xlabel('Regime')
        ax.set_ylabel('Annualized Return')
        ax.set_title('Returns by Regime')
        ax.grid(True, axis='y')
        
        # Add text annotations
        for i, v in enumerate(df['annualized_return']):
            ax.text(i, v + (0.01 if v >= 0 else -0.01), 
                   f"{v:.2%}\nVol: {df['annualized_volatility'].iloc[i]:.2%}", 
                   ha='center')
        
        return fig
    
    def plot_rolling_returns(self, returns: pd.Series, 
                           benchmark_returns: Optional[pd.Series] = None,
                           window: int = 252,
                           figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot rolling returns.
        
        Args:
            returns: Series of returns
            benchmark_returns: Series of benchmark returns
            window: Rolling window size
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Calculate rolling returns
        rolling_returns = returns.rolling(window=window).apply(
            lambda x: (1 + x).prod() - 1
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot rolling returns
        rolling_returns.plot(ax=ax, label='Strategy')
        
        # Plot benchmark rolling returns if provided
        if benchmark_returns is not None:
            rolling_benchmark = benchmark_returns.rolling(window=window).apply(
                lambda x: (1 + x).prod() - 1
            )
            rolling_benchmark.plot(ax=ax, label='Benchmark')
        
        # Add labels and title
        window_label = f"{window // self.annualization_factor}-Year" if window >= self.annualization_factor else f"{window}-Day"
        ax.set_xlabel('Date')
        ax.set_ylabel(f'Rolling {window_label} Return')
        ax.set_title(f'Rolling {window_label} Returns')
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def plot_rolling_volatility(self, returns: pd.Series, 
                              benchmark_returns: Optional[pd.Series] = None,
                              window: int = 63,
                              figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot rolling volatility.
        
        Args:
            returns: Series of returns
            benchmark_returns: Series of benchmark returns
            window: Rolling window size
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(self.annualization_factor)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot rolling volatility
        rolling_vol.plot(ax=ax, label='Strategy')
        
        # Plot benchmark rolling volatility if provided
        if benchmark_returns is not None:
            rolling_benchmark_vol = benchmark_returns.rolling(window=window).std() * np.sqrt(self.annualization_factor)
            rolling_benchmark_vol.plot(ax=ax, label='Benchmark')
        
        # Add labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Annualized Volatility')
        ax.set_title(f'Rolling {window}-Day Volatility')
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def plot_rolling_sharpe(self, returns: pd.Series, 
                          risk_free_rate: float,
                          window: int = 252,
                          figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot rolling Sharpe ratio.
        
        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate
            window: Rolling window size
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Calculate excess returns
        excess_returns = returns - risk_free_rate
        
        # Define rolling Sharpe function
        def rolling_sharpe(x):
            return x.mean() / x.std() * np.sqrt(self.annualization_factor)
        
        # Calculate rolling Sharpe ratio
        rolling_sharpe_ratio = excess_returns.rolling(window=window).apply(rolling_sharpe)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot rolling Sharpe ratio
        rolling_sharpe_ratio.plot(ax=ax)
        
        # Add horizontal line at 0
        ax.axhline(y=0, color='r', linestyle='--')
        
        # Add labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title(f'Rolling {window}-Day Sharpe Ratio')
        ax.grid(True)
        
        return fig
    
    def plot_underwater(self, returns: pd.Series, 
                       benchmark_returns: Optional[pd.Series] = None,
                       figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot underwater chart (drawdowns).
        
        Args:
            returns: Series of returns
            benchmark_returns: Series of benchmark returns
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Calculate drawdowns
        drawdowns = self._calculate_drawdowns(returns)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot drawdowns
        drawdowns.plot(ax=ax, label='Strategy')
        
        # Plot benchmark drawdowns if provided
        if benchmark_returns is not None:
            benchmark_drawdowns = self._calculate_drawdowns(benchmark_returns)
            benchmark_drawdowns.plot(ax=ax, label='Benchmark')
        
        # Add labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown')
        ax.set_title('Underwater Chart (Drawdowns)')
        ax.legend()
        ax.grid(True)
        
        # Invert y-axis for better visualization
        ax.invert_yaxis()
        
        return fig
    
    def _calculate_drawdowns(self, returns: pd.Series) -> pd.Series:
        """
        Calculate drawdowns.
        
        Args:
            returns: Series of returns
            
        Returns:
            Series of drawdowns
        """
        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative_returns.cummax()
        
        # Calculate drawdowns
        drawdowns = 1 - cumulative_returns / running_max
        
        return drawdowns


def calculate_turnover(old_weights: np.ndarray, new_weights: np.ndarray) -> float:
    """
    Calculate portfolio turnover.
    
    Args:
        old_weights: Old portfolio weights
        new_weights: New portfolio weights
        
    Returns:
        Turnover (sum of absolute changes)
    """
    return np.sum(np.abs(new_weights - old_weights))


def calculate_duration_contribution(weights: np.ndarray, durations: np.ndarray) -> np.ndarray:
    """
    Calculate duration contribution of each asset.
    
    Args:
        weights: Portfolio weights
        durations: Asset durations
        
    Returns:
        Duration contributions
    """
    return weights * durations


def calculate_tracking_error_contribution(weights: np.ndarray, 
                                         benchmark_weights: np.ndarray,
                                         covariance_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate tracking error contribution of each asset.
    
    Args:
        weights: Portfolio weights
        benchmark_weights: Benchmark weights
        covariance_matrix: Covariance matrix
        
    Returns:
        Tracking error contributions
    """
    # Calculate active weights
    active_weights = weights - benchmark_weights
    
    # Calculate tracking error
    tracking_variance = active_weights @ covariance_matrix @ active_weights
    tracking_error = np.sqrt(tracking_variance)
    
    # Calculate marginal contribution to tracking error
    mcte = covariance_matrix @ active_weights / tracking_error
    
    # Calculate contribution to tracking error
    cte = active_weights * mcte
    
    return cte


def calculate_risk_contribution(weights: np.ndarray, covariance_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate risk contribution of each asset.
    
    Args:
        weights: Portfolio weights
        covariance_matrix: Covariance matrix
        
    Returns:
        Risk contributions
    """
    # Calculate portfolio volatility
    portfolio_variance = weights @ covariance_matrix @ weights
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    # Calculate marginal contribution to risk
    mcr = covariance_matrix @ weights / portfolio_volatility
    
    # Calculate risk contribution
    rc = weights * mcr
    
    return rc
