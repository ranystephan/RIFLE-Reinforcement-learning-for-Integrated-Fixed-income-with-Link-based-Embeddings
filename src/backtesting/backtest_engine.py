"""
Backtesting Engine for Fixed Income RL Project

This module implements a backtesting framework for fixed income portfolio strategies:
1. Historical simulation of strategy performance
2. Benchmark comparison
3. Regime-conditional analysis
4. Transaction cost modeling
5. Duration-aware evaluation

Mathematical foundations:
- Return and risk metrics
- Performance attribution
- Rolling window analysis
- Regime-dependent performance

Author: ranycs & corentin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import logging
import time
from datetime import datetime, timedelta
import seaborn as sns
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Backtesting engine for fixed income portfolio strategies.
    """
    
    def __init__(self, 
                 market_data: Dict[str, pd.DataFrame],
                 bond_universe: pd.DataFrame,
                 initial_capital: float = 1000000.0,
                 rebalance_freq: int = 21,
                 transaction_cost: float = 0.0005,
                 risk_free_rate: float = 0.02,
                 benchmark_weights: Optional[np.ndarray] = None,
                 use_regimes: bool = True,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None):
        """
        Initialize backtesting engine.
        
        Args:
            market_data: Dictionary with market data ('rates', 'bond_prices', etc.)
            bond_universe: DataFrame with bond characteristics
            initial_capital: Initial capital
            rebalance_freq: Frequency of rebalancing in days
            transaction_cost: Transaction cost as a fraction of traded value
            risk_free_rate: Risk-free rate (annualized)
            benchmark_weights: Benchmark portfolio weights (if None, use equal weights)
            use_regimes: Whether to use regime information in the analysis
            start_date: Start date for backtest (if None, use first date in data)
            end_date: End date for backtest (if None, use last date in data)
        """
        self.market_data = market_data
        self.bond_universe = bond_universe
        self.initial_capital = initial_capital
        self.rebalance_freq = rebalance_freq
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate / 252  # Convert to daily
        self.use_regimes = use_regimes
        
        # Extract data
        self.rates_data = market_data['rates']
        self.bond_prices = market_data['bond_prices']
        
        # Set date range
        if start_date is not None:
            self.start_date = pd.to_datetime(start_date)
        else:
            self.start_date = self.rates_data.index[0]
        
        if end_date is not None:
            self.end_date = pd.to_datetime(end_date)
        else:
            self.end_date = self.rates_data.index[-1]
        
        # Filter data to date range - use integer indexing if dates are integers
        if isinstance(self.rates_data.index, pd.DatetimeIndex):
            # If index is DatetimeIndex, make sure our boundaries are datetime
            if not isinstance(self.start_date, pd.Timestamp):
                self.start_date = pd.to_datetime(self.start_date)
            if not isinstance(self.end_date, pd.Timestamp):
                self.end_date = pd.to_datetime(self.end_date)
            # Use datetime indexing
            self.rates_data = self.rates_data.loc[self.start_date:self.end_date]
            self.bond_prices = self.bond_prices.loc[self.start_date:self.end_date]
        else:
            # For non-datetime indices (like integers), use iloc
            start_idx = 0 if isinstance(self.start_date, pd.Timestamp) else int(self.start_date)
            end_idx = len(self.rates_data) if isinstance(self.end_date, pd.Timestamp) else int(self.end_date)
            self.rates_data = self.rates_data.iloc[start_idx:end_idx]
            self.bond_prices = self.bond_prices.iloc[start_idx:end_idx]
        
        # Number of assets
        self.num_assets = len(bond_universe)
        
        # Set benchmark weights
        if benchmark_weights is not None:
            self.benchmark_weights = benchmark_weights
        else:
            # Equal weight benchmark
            self.benchmark_weights = np.ones(self.num_assets) / self.num_assets
        
        # Initialize performance metrics
        self.strategy_returns = None
        self.benchmark_returns = None
        self.strategy_nav = None
        self.benchmark_nav = None
        self.strategy_weights = None
        self.metrics = {}
        self.regime_metrics = {}
    
    def run_backtest(self, strategy: Callable[[pd.DataFrame, Dict[str, Any]], np.ndarray],
                    strategy_args: Dict[str, Any] = {},
                    recompute_weights: bool = True) -> Dict[str, Any]:
        """
        Run backtest for the given strategy.
        
        Args:
            strategy: Function that takes market data and parameters and returns weights
            strategy_args: Arguments to pass to the strategy function
            recompute_weights: Whether to recompute weights at each rebalance date
                (False for using pre-computed weights)
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Running backtest from {self.start_date} to {self.end_date}")
        
        # Get dates
        dates = self.rates_data.index
        
        # Initialize portfolio
        portfolio_value = self.initial_capital
        portfolio_weights = self.benchmark_weights.copy()  # Start with benchmark
        portfolio_holdings = portfolio_value * portfolio_weights
        
        # Log initial state
        logger.info(f"Initial portfolio value: {portfolio_value}")
        logger.info(f"Number of assets: {self.num_assets}")
        logger.info(f"First few prices: {self.get_prices(dates[0])[:5]}")
        
        # Initialize arrays
        portfolio_values = np.zeros(len(dates))
        portfolio_returns = np.zeros(len(dates))
        portfolio_weights_history = np.zeros((len(dates), self.num_assets))
        
        # Benchmark
        benchmark_value = self.initial_capital
        benchmark_values = np.zeros(len(dates))
        benchmark_returns = np.zeros(len(dates))
        
        # Transaction costs
        transaction_costs = np.zeros(len(dates))
        
        # Rebalance dates
        days_since_rebalance = 0
        
        # Add diagnostic counters
        extreme_movement_count = 0
        rebalance_count = 0
        
        # Run backtest
        for i, date in enumerate(dates):
            # Get price changes
            if i > 0:
                prev_prices = self.get_prices(dates[i-1])
                current_prices = self.get_prices(date)
                
                # Calculate returns with safeguards against extreme movements
                # Add protection against extreme or invalid price returns
                price_returns = np.zeros_like(prev_prices)
                for j, (prev, curr) in enumerate(zip(prev_prices, current_prices)):
                    if prev <= 0 or curr <= 0:
                        # Invalid prices, assume no change
                        price_returns[j] = 0.0
                    else:
                        # Calculate return with reasonable bounds
                        raw_return = curr / prev - 1
                        # Cap daily returns to prevent extreme movements
                        price_returns[j] = np.clip(raw_return, -0.1, 0.1)
                
                # Update portfolio holdings with bounded returns
                portfolio_holdings = portfolio_holdings * (1 + price_returns)
                new_portfolio_value = np.sum(portfolio_holdings)
                
                # Calculate portfolio return with safety check
                if portfolio_value <= 0:
                    portfolio_return = 0  # Prevent division by zero
                else:
                    portfolio_return = (new_portfolio_value / portfolio_value) - 1
                
                portfolio_value = max(new_portfolio_value, 0.01 * self.initial_capital)  # Prevent zero/negative value
                
                # Update benchmark with same safeguards
                benchmark_holdings = benchmark_value * self.benchmark_weights
                benchmark_holdings = benchmark_holdings * (1 + price_returns)
                new_benchmark_value = np.sum(benchmark_holdings)
                
                if benchmark_value <= 0:
                    benchmark_return = 0
                else:
                    benchmark_return = (new_benchmark_value / benchmark_value) - 1
                
                benchmark_value = max(new_benchmark_value, 0.01 * self.initial_capital)  # Prevent zero/negative value
            else:
                # First day, no returns
                portfolio_return = 0
                benchmark_return = 0
            
            # Store values and returns
            portfolio_values[i] = portfolio_value
            portfolio_returns[i] = portfolio_return
            benchmark_values[i] = benchmark_value
            benchmark_returns[i] = benchmark_return
            
            # Update portfolio weights
            portfolio_weights = portfolio_holdings / portfolio_value
            portfolio_weights_history[i] = portfolio_weights
            
            # Increment days since rebalance
            days_since_rebalance += 1
            
            # Rebalance if needed
            if days_since_rebalance >= self.rebalance_freq:
                # Get new weights from strategy
                if recompute_weights:
                    # Prepare market data for strategy
                    market_snapshot = self._prepare_market_snapshot(date)
                    
                    # Call strategy function
                    try:
                        new_weights = strategy(market_snapshot, strategy_args)
                        
                        # Ensure weights are valid
                        new_weights = np.array(new_weights).astype(float)
                        if np.isnan(new_weights).any() or np.sum(new_weights) == 0:
                            logger.warning(f"Invalid weights at {date}. Using current weights.")
                            new_weights = portfolio_weights
                        else:
                            # Normalize weights to sum to 1
                            new_weights = new_weights / np.sum(new_weights)
                    except Exception as e:
                        logger.error(f"Error in strategy at {date}: {e}")
                        new_weights = portfolio_weights
                else:
                    # Use pre-computed weights
                    time_idx = i
                    if 'weights' in strategy_args and time_idx < len(strategy_args['weights']):
                        new_weights = strategy_args['weights'][time_idx]
                    else:
                        logger.warning(f"No pre-computed weights at {date}. Using current weights.")
                        new_weights = portfolio_weights
                
                # Calculate turnover
                turnover = np.sum(np.abs(portfolio_weights - new_weights))
                
                # Apply transaction costs
                transaction_cost = turnover * self.transaction_cost
                portfolio_value *= (1 - transaction_cost)
                transaction_costs[i] = transaction_cost
                
                # Update weights and holdings
                portfolio_weights = new_weights
                portfolio_holdings = portfolio_value * portfolio_weights
                
                # Reset rebalance counter
                days_since_rebalance = 0
                
                # Increment rebalance counter
                rebalance_count += 1
            
            # Check for extreme movements
            if np.abs(portfolio_return) > 0.1 or np.abs(benchmark_return) > 0.1:
                extreme_movement_count += 1
        
        # Calculate metrics
        self.strategy_returns = pd.Series(portfolio_returns, index=dates)
        self.benchmark_returns = pd.Series(benchmark_returns, index=dates)
        self.strategy_nav = pd.Series(portfolio_values, index=dates)
        self.benchmark_nav = pd.Series(benchmark_values, index=dates)
        self.strategy_weights = pd.DataFrame(portfolio_weights_history, index=dates)
        self.transaction_costs = pd.Series(transaction_costs, index=dates)
        
        # Calculate performance metrics
        self.metrics = self.calculate_performance_metrics(
            self.strategy_returns,
            self.benchmark_returns,
            self.risk_free_rate
        )
        
        # Calculate regime-dependent metrics
        if self.use_regimes and 'regime' in self.rates_data.columns:
            self.regime_metrics = self.calculate_regime_metrics(
                self.strategy_returns,
                self.benchmark_returns,
                self.rates_data['regime']
            )
        
        # Log diagnostics
        logger.info(f"Backtest diagnostics:")
        logger.info(f"  Total rebalances: {rebalance_count}")
        logger.info(f"  Extreme movements: {extreme_movement_count}")
        logger.info(f"  Final portfolio value: {portfolio_value:.2f}")
        logger.info(f"  Total return: {self.metrics['total_return']:.4f}")
        logger.info(f"  Average return: {self.strategy_returns.mean():.6f}")
        logger.info(f"  Return std dev: {self.strategy_returns.std():.6f}")
        logger.info(f"  Positive returns: {(self.strategy_returns > 0).mean():.4f}")
        logger.info(f"  Min return: {self.strategy_returns.min():.4f}")
        logger.info(f"  Max return: {self.strategy_returns.max():.4f}")
        
        # Prepare results
        results = {
            'metrics': self.metrics,
            'regime_metrics': self.regime_metrics,
            'strategy_returns': self.strategy_returns,
            'benchmark_returns': self.benchmark_returns,
            'strategy_nav': self.strategy_nav,
            'benchmark_nav': self.benchmark_nav,
            'strategy_weights': self.strategy_weights,
            'transaction_costs': self.transaction_costs
        }
        
        logger.info(f"Backtest completed with Sharpe ratio: {self.metrics['sharpe_ratio']:.2f}")
        
        return results
    
    def get_prices(self, date: Union[str, pd.Timestamp]) -> np.ndarray:
        """
        Get bond prices for a specific date.
        
        Args:
            date: Date to get prices for
            
        Returns:
            Array of prices
        """
        # Extract price columns
        price_cols = [col for col in self.bond_prices.columns if 'price_' in col]
        
        # Get prices for date
        prices = self.bond_prices.loc[date, price_cols].values
        
        return prices
    
    def _prepare_market_snapshot(self, date: Union[str, pd.Timestamp]) -> Dict[str, Any]:
        """
        Prepare market data snapshot for strategy.
        
        Args:
            date: Date to prepare snapshot for
            
        Returns:
            Dictionary with market data snapshot
        """
        # Get date index
        date_idx = self.rates_data.index.get_loc(date)
        
        # Get historical data up to date
        rates_history = self.rates_data.iloc[:date_idx+1].copy()
        prices_history = self.bond_prices.iloc[:date_idx+1].copy()
        
        # Create snapshot
        snapshot = {
            'current_date': date,
            'rates': rates_history,
            'prices': prices_history,
            'bond_universe': self.bond_universe,
            'current_rates': rates_history.iloc[-1],
            'current_prices': prices_history.iloc[-1]
        }
        
        return snapshot
    
    @staticmethod
    def calculate_performance_metrics(strategy_returns: pd.Series,
                                     benchmark_returns: pd.Series,
                                     risk_free_rate: float) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            strategy_returns: Series of strategy returns
            benchmark_returns: Series of benchmark returns
            risk_free_rate: Risk-free rate (daily)
            
        Returns:
            Dictionary with performance metrics
        """
        # Handle NaN values
        strategy_returns = strategy_returns.fillna(0)
        benchmark_returns = benchmark_returns.fillna(0)
        
        # Calculate basic metrics
        total_return = (1 + strategy_returns).prod() - 1
        benchmark_total_return = (1 + benchmark_returns).prod() - 1
        
        # Annualized return
        num_years = len(strategy_returns) / 252
        annualized_return = (1 + total_return) ** (1 / num_years) - 1
        benchmark_annualized_return = (1 + benchmark_total_return) ** (1 / num_years) - 1
        
        # Volatility
        volatility = strategy_returns.std() * np.sqrt(252)
        benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        excess_returns = strategy_returns - risk_free_rate
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Sortino ratio
        downside_returns = excess_returns[excess_returns < 0]
        sortino_ratio = excess_returns.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + strategy_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdowns = 1 - cumulative_returns / running_max
        max_drawdown = drawdowns.max() if not np.isnan(drawdowns.max()) else 0
        
        # Information ratio
        active_returns = strategy_returns - benchmark_returns
        information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(252) if active_returns.std() > 0 else np.nan
        
        # Tracking error
        tracking_error = active_returns.std() * np.sqrt(252)
        
        # Batting average
        batting_average = (active_returns > 0).mean()
        
        # Win/loss ratio
        wins = active_returns[active_returns > 0]
        losses = active_returns[active_returns < 0]
        win_loss_ratio = abs(wins.mean() / losses.mean()) if len(losses) > 0 and losses.mean() != 0 else np.nan
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else np.nan
        
        # Metrics dictionary
        metrics = {
            'total_return': total_return,
            'benchmark_total_return': benchmark_total_return,
            'annualized_return': annualized_return,
            'benchmark_annualized_return': benchmark_annualized_return,
            'volatility': volatility,
            'benchmark_volatility': benchmark_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'information_ratio': information_ratio,
            'tracking_error': tracking_error,
            'batting_average': batting_average,
            'win_loss_ratio': win_loss_ratio,
            'calmar_ratio': calmar_ratio
        }
        
        return metrics
    
    @staticmethod
    def calculate_regime_metrics(strategy_returns: pd.Series,
                                benchmark_returns: pd.Series,
                                regimes: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Calculate performance metrics by regime.
        
        Args:
            strategy_returns: Series of strategy returns
            benchmark_returns: Series of benchmark returns
            regimes: Series of regime labels
            
        Returns:
            Dictionary with performance metrics by regime
        """
        # Get unique regimes
        unique_regimes = regimes.unique()
        
        # Initialize metrics dictionary
        regime_metrics = {}
        
        # Calculate metrics for each regime
        for regime in unique_regimes:
            # Filter returns by regime
            regime_filter = regimes == regime
            regime_strategy_returns = strategy_returns[regime_filter]
            regime_benchmark_returns = benchmark_returns[regime_filter]
            
            # Calculate alpha and beta
            if len(regime_strategy_returns) > 0:
                # Calculate returns and volatility
                regime_total_return = (1 + regime_strategy_returns).prod() - 1
                regime_benchmark_total_return = (1 + regime_benchmark_returns).prod() - 1
                
                regime_volatility = regime_strategy_returns.std() * np.sqrt(252)
                regime_benchmark_volatility = regime_benchmark_returns.std() * np.sqrt(252)
                
                # Calculate alpha and beta
                cov = np.cov(regime_strategy_returns, regime_benchmark_returns)[0, 1]
                var = np.var(regime_benchmark_returns)
                beta = cov / var if var > 0 else np.nan
                
                # Calculate average return
                avg_return = regime_strategy_returns.mean() * 252
                avg_benchmark_return = regime_benchmark_returns.mean() * 252
                
                # Calculate alpha
                alpha = avg_return - beta * avg_benchmark_return
                
                # Calculate information ratio
                active_returns = regime_strategy_returns - regime_benchmark_returns
                information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(252) if active_returns.std() > 0 else np.nan
                
                # Store metrics
                regime_metrics[str(regime)] = {
                    'total_return': regime_total_return,
                    'benchmark_total_return': regime_benchmark_total_return,
                    'volatility': regime_volatility,
                    'benchmark_volatility': regime_benchmark_volatility,
                    'alpha': alpha,
                    'beta': beta,
                    'information_ratio': information_ratio,
                    'num_days': len(regime_strategy_returns)
                }
        
        return regime_metrics
    
    def plot_performance(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot cumulative performance of strategy vs. benchmark.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.strategy_nav is None or self.benchmark_nav is None:
            raise ValueError("Run backtest first")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot NAVs
        self.strategy_nav.plot(label='Strategy', ax=ax)
        self.benchmark_nav.plot(label='Benchmark', ax=ax)
        
        # Add labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.set_title('Strategy vs. Benchmark Performance')
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def plot_returns(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot returns of strategy vs. benchmark.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.strategy_returns is None or self.benchmark_returns is None:
            raise ValueError("Run backtest first")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot returns
        self.strategy_returns.plot(label='Strategy', ax=ax)
        self.benchmark_returns.plot(label='Benchmark', ax=ax)
        
        # Add labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Return')
        ax.set_title('Strategy vs. Benchmark Returns')
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def plot_drawdowns(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot drawdowns of strategy vs. benchmark.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.strategy_returns is None or self.benchmark_returns is None:
            raise ValueError("Run backtest first")
        
        # Calculate drawdowns
        strategy_cumulative = (1 + self.strategy_returns).cumprod()
        benchmark_cumulative = (1 + self.benchmark_returns).cumprod()
        
        strategy_drawdowns = 1 - strategy_cumulative / strategy_cumulative.cummax()
        benchmark_drawdowns = 1 - benchmark_cumulative / benchmark_cumulative.cummax()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot drawdowns
        strategy_drawdowns.plot(label='Strategy', ax=ax)
        benchmark_drawdowns.plot(label='Benchmark', ax=ax)
        
        # Add labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown')
        ax.set_title('Strategy vs. Benchmark Drawdowns')
        ax.legend()
        ax.grid(True)
        ax.invert_yaxis()  # Invert y-axis for better visualization
        
        return fig
    
    def plot_weights(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot portfolio weights over time.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.strategy_weights is None:
            raise ValueError("Run backtest first")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot weights as area chart
        self.strategy_weights.plot.area(ax=ax)
        
        # Add labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Weight')
        ax.set_title('Portfolio Weights Over Time')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(True)
        
        return fig
    
    def plot_regime_performance(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot performance by regime.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.regime_metrics:
            raise ValueError("Run backtest with use_regimes=True first")
        
        # Extract metrics
        regimes = list(self.regime_metrics.keys())
        returns = [self.regime_metrics[r]['total_return'] for r in regimes]
        benchmark_returns = [self.regime_metrics[r]['benchmark_total_return'] for r in regimes]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Bar width
        width = 0.35
        
        # Set up indices
        indices = np.arange(len(regimes))
        
        # Plot bars
        ax.bar(indices - width/2, returns, width, label='Strategy')
        ax.bar(indices + width/2, benchmark_returns, width, label='Benchmark')
        
        # Add labels and title
        ax.set_xlabel('Regime')
        ax.set_ylabel('Total Return')
        ax.set_title('Performance by Regime')
        ax.set_xticks(indices)
        ax.set_xticklabels(regimes)
        ax.legend()
        ax.grid(True, axis='y')
        
        return fig
    
    def plot_metrics_summary(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot summary of performance metrics.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.metrics:
            raise ValueError("Run backtest first")
        
        # Select metrics to display
        metrics_to_plot = [
            'annualized_return',
            'volatility',
            'sharpe_ratio',
            'sortino_ratio',
            'max_drawdown',
            'information_ratio',
            'tracking_error'
        ]
        
        # Extract values
        strategy_values = [self.metrics[m] for m in metrics_to_plot]
        benchmark_values = [self.metrics[f'benchmark_{m}'] if f'benchmark_{m}' in self.metrics else np.nan 
                           for m in metrics_to_plot]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Bar width
        width = 0.35
        
        # Set up indices
        indices = np.arange(len(metrics_to_plot))
        
        # Plot bars
        ax.bar(indices - width/2, strategy_values, width, label='Strategy')
        ax.bar(indices + width/2, benchmark_values, width, label='Benchmark')
        
        # Add labels and title
        ax.set_xlabel('Metric')
        ax.set_ylabel('Value')
        ax.set_title('Performance Metrics Summary')
        ax.set_xticks(indices)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_to_plot])
        ax.legend()
        ax.grid(True, axis='y')
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def generate_report(self, filename: str):
        """
        Generate a comprehensive performance report as HTML.
        
        Args:
            filename: Filename for the report
        """
        import pandas as pd
        from IPython.display import HTML, display
        
        if not self.metrics:
            raise ValueError("Run backtest first")
        
        # Create HTML content
        html_content = f"""
        <html>
        <head>
            <title>Backtest Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                }}
                h1, h2 {{
                    color: #333;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 20px;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .metrics-table {{
                    width: 60%;
                }}
                .regime-table {{
                    width: 80%;
                }}
            </style>
        </head>
        <body>
            <h1>Fixed Income Portfolio Backtest Report</h1>
            <p>Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}</p>
            
            <h2>Performance Summary</h2>
            <table class="metrics-table">
                <tr>
                    <th>Metric</th>
                    <th>Strategy</th>
                    <th>Benchmark</th>
                </tr>
                <tr>
                    <td>Total Return</td>
                    <td>{self.metrics['total_return']:.2%}</td>
                    <td>{self.metrics['benchmark_total_return']:.2%}</td>
                </tr>
                <tr>
                    <td>Annualized Return</td>
                    <td>{self.metrics['annualized_return']:.2%}</td>
                    <td>{self.metrics['benchmark_annualized_return']:.2%}</td>
                </tr>
                <tr>
                    <td>Volatility</td>
                    <td>{self.metrics['volatility']:.2%}</td>
                    <td>{self.metrics['benchmark_volatility']:.2%}</td>
                </tr>
                <tr>
                    <td>Sharpe Ratio</td>
                    <td>{self.metrics['sharpe_ratio']:.2f}</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Sortino Ratio</td>
                    <td>{self.metrics['sortino_ratio']:.2f}</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Maximum Drawdown</td>
                    <td>{self.metrics['max_drawdown']:.2%}</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Information Ratio</td>
                    <td>{self.metrics['information_ratio']:.2f}</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Tracking Error</td>
                    <td>{self.metrics['tracking_error']:.2%}</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Batting Average</td>
                    <td>{self.metrics['batting_average']:.2f}</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Win/Loss Ratio</td>
                    <td>{self.metrics['win_loss_ratio']:.2f}</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Calmar Ratio</td>
                    <td>{self.metrics['calmar_ratio']:.2f}</td>
                    <td>-</td>
                </tr>
            </table>
        """
        
        # Add regime performance if available
        if self.regime_metrics:
            html_content += """
            <h2>Performance by Regime</h2>
            <table class="regime-table">
                <tr>
                    <th>Regime</th>
                    <th>Days</th>
                    <th>Strategy Return</th>
                    <th>Benchmark Return</th>
                    <th>Alpha</th>
                    <th>Beta</th>
                    <th>Information Ratio</th>
                </tr>
            """
            
            for regime, metrics in self.regime_metrics.items():
                html_content += f"""
                <tr>
                    <td>{regime}</td>
                    <td>{metrics['num_days']}</td>
                    <td>{metrics['total_return']:.2%}</td>
                    <td>{metrics['benchmark_total_return']:.2%}</td>
                    <td>{metrics['alpha']:.2%}</td>
                    <td>{metrics['beta']:.2f}</td>
                    <td>{metrics['information_ratio']:.2f}</td>
                </tr>
                """
            
            html_content += "</table>"
        
        # Close HTML
        html_content += """
        </body>
        </html>
        """
        
        # Write to file
        with open(filename, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Report generated and saved to {filename}")
        
        # Return HTML for display in notebook
        return HTML(html_content)


class RegimeClassifierStrategy:
    """
    Strategy based on regime classification with pre-defined weights per regime.
    """
    
    def __init__(self, 
                 regime_detector,
                 regime_weights: Dict[int, np.ndarray],
                 default_weights: Optional[np.ndarray] = None,
                 lookback_window: int = 21):
        """
        Initialize regime classifier strategy.
        
        Args:
            regime_detector: Regime detector model
            regime_weights: Dictionary mapping regime indices to portfolio weights
            default_weights: Default weights to use if regime not found
            lookback_window: Lookback window for regime detection
        """
        self.regime_detector = regime_detector
        self.regime_weights = regime_weights
        self.default_weights = default_weights
        self.lookback_window = lookback_window
    
    def __call__(self, market_snapshot: Dict[str, Any], args: Dict[str, Any]) -> np.ndarray:
        """
        Generate portfolio weights based on current market regime.
        
        Args:
            market_snapshot: Market data snapshot
            args: Additional arguments
            
        Returns:
            Portfolio weights
        """
        # Get rates data
        rates_data = market_snapshot['rates']
        
        # Get current date
        current_date = market_snapshot['current_date']
        
        # Detect regime
        if 'regime' in rates_data.columns:
            # Use pre-computed regime
            current_regime = rates_data.loc[current_date, 'regime']
        else:
            # Detect regime
            lookback_data = rates_data.iloc[-self.lookback_window:]
            regime_features = self.regime_detector.prepare_regime_detection_features(lookback_data)
            current_regime = self.regime_detector.predict(regime_features)[-1]
        
        # Get weights for regime
        if current_regime in self.regime_weights:
            weights = self.regime_weights[current_regime]
        elif self.default_weights is not None:
            weights = self.default_weights
        else:
            # Use equal weights
            num_assets = len(market_snapshot['bond_universe'])
            weights = np.ones(num_assets) / num_assets
        
        return weights


class DurationTargetingStrategy:
    """
    Strategy that targets specific portfolio duration based on market regimes.
    """
    
    def __init__(self, 
                 regime_detector,
                 regime_durations: Dict[int, float],
                 default_duration: float = 5.0,
                 lookback_window: int = 21,
                 portfolio_optimizer = None):
        """
        Initialize duration targeting strategy.
        
        Args:
            regime_detector: Regime detector model
            regime_durations: Dictionary mapping regime indices to target durations
            default_duration: Default duration to use if regime not found
            lookback_window: Lookback window for regime detection
            portfolio_optimizer: Portfolio optimizer instance
        """
        self.regime_detector = regime_detector
        self.regime_durations = regime_durations
        self.default_duration = default_duration
        self.lookback_window = lookback_window
        
        # Import portfolio optimizer if not provided
        if portfolio_optimizer is None:
            from src.models.portfolio_optimizer import FixedIncomePortfolioOptimizer
            self.optimizer = FixedIncomePortfolioOptimizer()
        else:
            self.optimizer = portfolio_optimizer
    
    def __call__(self, market_snapshot: Dict[str, Any], args: Dict[str, Any]) -> np.ndarray:
        """
        Generate portfolio weights based on target duration for current market regime.
        
        Args:
            market_snapshot: Market data snapshot
            args: Additional arguments
            
        Returns:
            Portfolio weights
        """
        # Get rates data
        rates_data = market_snapshot['rates']
        prices_data = market_snapshot['prices']
        
        # Get current date
        current_date = market_snapshot['current_date']
        
        # Detect regime
        if 'regime' in rates_data.columns:
            # Use pre-computed regime
            current_regime = rates_data.loc[current_date, 'regime']
        else:
            # Detect regime
            lookback_data = rates_data.iloc[-self.lookback_window:]
            regime_features = self.regime_detector.prepare_regime_detection_features(lookback_data)
            current_regime = self.regime_detector.predict(regime_features)[-1]
        
        # Get target duration for regime
        if current_regime in self.regime_durations:
            target_duration = self.regime_durations[current_regime]
        else:
            target_duration = self.default_duration
        
        # Get current durations
        duration_cols = [col for col in prices_data.columns if 'duration_' in col]
        durations = prices_data.loc[current_date, duration_cols].values
        
        # Estimate expected returns
        # For simplicity, use historical returns
        return_cols = [col for col in prices_data.columns if 'price_' in col]
        price_history = prices_data.loc[:current_date, return_cols].iloc[-63:]  # ~3 months
        returns_history = price_history.pct_change().dropna()
        expected_returns = returns_history.mean().values
        
        # Estimate covariance matrix
        covariance_matrix = returns_history.cov().values
        
        # Optimize portfolio to target duration
        optimal_weights = self.optimizer.duration_matching_optimization(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            durations=durations,
            target_duration=target_duration,
            constraints={'long_only': True}
        )
        
        if optimal_weights is None:
            # Fallback to equal weights
            optimal_weights = np.ones(len(durations)) / len(durations)
        
        return optimal_weights


class RLAgentStrategy:
    """
    Strategy based on a trained reinforcement learning agent.
    """
    
    def __init__(self, agent, state_preprocessor: Callable = None):
        """
        Initialize RL agent strategy.
        
        Args:
            agent: Trained RL agent
            state_preprocessor: Function to preprocess state for agent
        """
        self.agent = agent
        self.state_preprocessor = state_preprocessor
    
    def __call__(self, market_snapshot: Dict[str, Any], args: Dict[str, Any]) -> np.ndarray:
        """
        Generate portfolio weights using the trained RL agent.
        
        Args:
            market_snapshot: Market data snapshot
            args: Additional arguments
            
        Returns:
            Portfolio weights
        """
        # Preprocess state
        if self.state_preprocessor:
            state = self.state_preprocessor(market_snapshot)
        else:
            # Default preprocessing
            state = self._default_state_preprocessor(market_snapshot)
        
        # Get action from agent (eval mode)
        agent_weights = self.agent.select_action(state, eval_mode=True)
        
        # Handle dimension mismatch between agent output and bond universe
        num_assets = len(market_snapshot['bond_universe'])
        if len(agent_weights) != num_assets:
            # Create a full weights array with zeros
            weights = np.zeros(num_assets)
            
            # Assign agent weights to the first n positions
            n = min(len(agent_weights), num_assets)
            weights[:n] = agent_weights[:n]
            
            # Re-normalize to ensure weights sum to 1
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                # Fallback to equal weights if all weights are zero
                weights = np.ones(num_assets) / num_assets
                
            return weights
        
        return agent_weights
    
    def _default_state_preprocessor(self, market_snapshot: Dict[str, Any]) -> np.ndarray:
        """
        Default state preprocessor function.
        
        Args:
            market_snapshot: Market data snapshot
            
        Returns:
            Processed state vector
        """
        # Get rates data
        rates_data = market_snapshot['rates']
        current_date = market_snapshot['current_date']
        
        # Get price data
        prices_data = market_snapshot['prices']
        
        # Extract features
        # 1. Market features
        market_features = []
        
        # Interest rates
        if 'short_rate' in rates_data.columns:
            short_rate = rates_data.loc[current_date, 'short_rate']
            market_features.append(short_rate)
        else:
            market_features.append(0.0)
        
        # Yield curve slope
        if 'slope' in rates_data.columns:
            slope = rates_data.loc[current_date, 'slope']
            market_features.append(slope)
        elif '2s10s_spread' in rates_data.columns:
            slope = rates_data.loc[current_date, '2s10s_spread']
            market_features.append(slope)
        else:
            market_features.append(0.0)
        
        # Credit spreads
        for rating in ['AAA', 'BBB', 'B']:
            col = f"spread_{rating}"
            if col in rates_data.columns:
                spread = rates_data.loc[current_date, col]
                market_features.append(spread)
            else:
                market_features.append(0.0)
        
        # Regime one-hot encoding
        if 'regime' in rates_data.columns:
            regime = rates_data.loc[current_date, 'regime']
            num_regimes = len(rates_data['regime'].unique())
            regime_one_hot = np.zeros(num_regimes)
            regime_one_hot[regime] = 1
            market_features.extend(regime_one_hot)
        else:
            # Add placeholder zeros
            market_features.extend([0.0] * 4)  # Default 4 regimes
        
        # 2. Extract asset features
        num_assets = len(market_snapshot['bond_universe'])
        
        # Equal weights for placeholder
        portfolio_weights = np.ones(num_assets) / num_assets
        
        # Duration
        duration_cols = [col for col in prices_data.columns if 'duration_' in col]
        asset_durations = prices_data.loc[current_date, duration_cols].values
        
        # Convexity
        convexity_cols = [col for col in prices_data.columns if 'convexity_' in col]
        asset_convexities = prices_data.loc[current_date, convexity_cols].values
        
        # 3. Historical returns
        price_cols = [col for col in prices_data.columns if 'price_' in col]
        price_history = prices_data.loc[:current_date, price_cols].iloc[-63:]  # ~3 months
        returns_history = price_history.pct_change().dropna().values.flatten()
        
        # Combine features
        state = np.concatenate([
            market_features,
            portfolio_weights,
            asset_durations,
            asset_convexities,
            returns_history
        ])
        
        return state
