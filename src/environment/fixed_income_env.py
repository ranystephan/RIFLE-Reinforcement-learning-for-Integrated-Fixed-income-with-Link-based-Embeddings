"""
Fixed Income RL Environment Module

This module implements a Gymnasium environment for fixed income portfolio management:
1. State space definition (market features, portfolio characteristics)
2. Action space definition (portfolio weights, duration targets)
3. Step dynamics (portfolio rebalancing, market evolution)
4. Reward function (risk-adjusted returns, constraint penalties)

Mathematical foundations:
- MDP formulation for portfolio optimization
- Dynamic portfolio metrics (Sharpe ratio, tracking error)
- Market regime conditioning

Author: ranycs & cosrv
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
from src.environment.reward_function import create_reward_function, RewardFunction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FixedIncomeEnv(gym.Env):
    """
    Reinforcement learning environment for fixed income portfolio management.
    
    This environment simulates a fixed income portfolio manager's task:
    - Observing market states (yield curves, credit spreads, economic indicators)
    - Making portfolio allocation decisions
    - Receiving rewards based on risk-adjusted returns
    
    The state space includes:
    - Current portfolio characteristics (weights, duration, etc.)
    - Market features (interest rates, credit spreads, regimes)
    - Economic indicators
    
    The action space represents:
    - Target asset allocations
    - OR Duration/convexity targets
    - OR Sector rotation decisions
    """
    
    def __init__(self, 
                market_data: Dict[str, pd.DataFrame],
                bond_universe: pd.DataFrame,
                num_assets: int,
                initial_cash: float = 1000000.0,
                rebalance_freq: int = 21,
                transaction_cost: float = 0.0005,
                reward_type: str = 'sharpe',
                reward_function: Optional[RewardFunction] = None,
                regime_aware: bool = True,
                duration_constraint: Optional[Tuple[float, float]] = None,
                random_start: bool = True,
                window_size: int = 63,
                risk_free_rate: float = 0.0):
        """
        Initialize the fixed income environment.
        
        Args:
            market_data: Dictionary with market data ('rates', 'bond_prices', etc.)
            bond_universe: DataFrame with bond characteristics
            num_assets: Number of assets to include in the portfolio
            initial_cash: Initial cash amount
            rebalance_freq: Frequency of rebalancing in days
            transaction_cost: Transaction cost as a fraction of traded value
            reward_type: Type of reward function ('sharpe', 'returns', 'risk_adjusted')
            reward_function: Optional, an instantiated reward function object to use
            regime_aware: Whether to include regime information in the state
            duration_constraint: Tuple of (min_duration, max_duration)
            random_start: Whether to start at a random point in the data
            window_size: Size of the lookback window for features
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        super(FixedIncomeEnv, self).__init__()
        
        # Store parameters
        self.market_data = market_data
        self.bond_universe = bond_universe
        self.num_assets = min(num_assets, len(bond_universe))
        self.initial_cash = initial_cash
        self.rebalance_freq = rebalance_freq
        self.transaction_cost = transaction_cost
        self.reward_type = reward_type
        self.regime_aware = regime_aware
        self.duration_constraint = duration_constraint
        self.random_start = random_start
        self.window_size = window_size
        self.risk_free_rate = risk_free_rate
        
        # Create or store reward function
        if reward_function is None:
            self.reward_function = create_reward_function(reward_type, risk_free_rate=risk_free_rate)
        else:
            self.reward_function = reward_function
        
        # Extract data
        self.rates_data = market_data['rates']
        self.bond_prices = market_data['bond_prices']
        
        # Select assets for the universe
        # For simplicity, we'll select the first num_assets
        # In a real application, this would be a more sophisticated selection
        self.selected_assets = bond_universe.index[:self.num_assets].tolist()
        
        # Create features for selected assets
        self.create_asset_features()
        
        # Calculate total number of steps
        self.max_steps = len(self.rates_data) - window_size - 1
        
        # Define observation space
        self._define_observation_space()
        
        # Define action space (portfolio weights)
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.num_assets,), dtype=np.float32
        )
        
        # Reset the environment
        self.reset()
    
    def create_asset_features(self):
        """
        Create features for the selected assets.
        """
        logger.info("Creating asset features")
        
        # Filter bond universe to selected assets
        self.asset_features = self.bond_universe.loc[self.selected_assets].copy()
        
        # Log selected assets and available columns for debugging
        logger.info(f"Selected assets: {self.selected_assets}")
        logger.info(f"Available bond price columns: {self.bond_prices.columns.tolist()[:10]}...")
        
        # Construct expected column patterns
        price_patterns = [f"price_BOND{asset:04d}" if isinstance(asset, int) else f"price_{asset}" for asset in self.selected_assets]
        duration_patterns = [f"duration_BOND{asset:04d}" if isinstance(asset, int) else f"duration_{asset}" for asset in self.selected_assets]
        convexity_patterns = [f"convexity_BOND{asset:04d}" if isinstance(asset, int) else f"convexity_{asset}" for asset in self.selected_assets]
        
        # Filter bond prices to selected assets - using more flexible matching to handle naming differences
        price_cols = []
        duration_cols = []
        convexity_cols = []
        
        for col in self.bond_prices.columns:
            # Check for price columns
            if any(pattern in col for pattern in price_patterns):
                price_cols.append(col)
            # Check for duration columns
            elif any(pattern in col for pattern in duration_patterns):
                duration_cols.append(col)
            # Check for convexity columns
            elif any(pattern in col for pattern in convexity_patterns):
                convexity_cols.append(col)
        
        logger.info(f"Found {len(price_cols)} price columns, {len(duration_cols)} duration columns, and {len(convexity_cols)} convexity columns")
        
        # Handle case where we don't have enough data
        if len(price_cols) < self.num_assets:
            logger.warning(f"Not enough price data found. Expected {self.num_assets} assets, found {len(price_cols)}")
            
        # If no duration or convexity data is found, create synthetic ones
        if len(duration_cols) == 0:
            logger.warning("No duration data found in bond prices. Creating synthetic duration data.")
            # Create synthetic durations between 1 and 7 years
            duration_data = np.random.uniform(1.0, 7.0, size=(len(self.bond_prices), self.num_assets))
            self.asset_durations = pd.DataFrame(
                duration_data, 
                index=self.bond_prices.index,
                columns=[f"duration_{i}" for i in range(self.num_assets)]
            )
        else:
            self.asset_durations = self.bond_prices[duration_cols]
            # Rename columns for simplicity
            self.asset_durations.columns = [f"duration_{i}" for i in range(len(duration_cols))]
        
        if len(convexity_cols) == 0:
            logger.warning("No convexity data found in bond prices. Creating synthetic convexity data.")
            # Create synthetic convexities based on durations (approximately duration^2 / 100)
            if hasattr(self, 'asset_durations'):
                convexity_data = np.square(self.asset_durations.values) / 100
            else:
                convexity_data = np.random.uniform(0.1, 1.0, size=(len(self.bond_prices), self.num_assets))
            
            self.asset_convexities = pd.DataFrame(
                convexity_data,
                index=self.bond_prices.index,
                columns=[f"convexity_{i}" for i in range(self.num_assets)]
            )
        else:
            self.asset_convexities = self.bond_prices[convexity_cols]
            # Rename columns for simplicity
            self.asset_convexities.columns = [f"convexity_{i}" for i in range(len(convexity_cols))]
        
        # Process price data
        if len(price_cols) > 0:
            self.asset_prices = self.bond_prices[price_cols]
            # Rename columns for simplicity
            self.asset_prices.columns = [f"price_{i}" for i in range(len(price_cols))]
        else:
            logger.warning("No price data found in bond prices. Creating synthetic price data.")
            # Create synthetic prices around 100 (par)
            price_data = np.random.uniform(95.0, 105.0, size=(len(self.bond_prices), self.num_assets))
            self.asset_prices = pd.DataFrame(
                price_data,
                index=self.bond_prices.index,
                columns=[f"price_{i}" for i in range(self.num_assets)]
            )
        
        # Ensure all dataframes have the same number of columns
        expected_cols = self.num_assets
        
        # Validate or adjust dimensions
        if self.asset_prices.shape[1] != expected_cols:
            logger.warning(f"Asset prices shape mismatch. Expected {expected_cols} columns, got {self.asset_prices.shape[1]}.")
            # Pad with synthetic data if needed
            if self.asset_prices.shape[1] < expected_cols:
                missing = expected_cols - self.asset_prices.shape[1]
                synthetic = pd.DataFrame(
                    np.random.uniform(95.0, 105.0, size=(len(self.bond_prices), missing)),
                    index=self.bond_prices.index,
                    columns=[f"price_{i+self.asset_prices.shape[1]}" for i in range(missing)]
                )
                self.asset_prices = pd.concat([self.asset_prices, synthetic], axis=1)
        
        if self.asset_durations.shape[1] != expected_cols:
            logger.warning(f"Asset durations shape mismatch. Expected {expected_cols} columns, got {self.asset_durations.shape[1]}.")
            # Pad with synthetic data if needed
            if self.asset_durations.shape[1] < expected_cols:
                missing = expected_cols - self.asset_durations.shape[1]
                synthetic = pd.DataFrame(
                    np.random.uniform(1.0, 7.0, size=(len(self.bond_prices), missing)),
                    index=self.bond_prices.index,
                    columns=[f"duration_{i+self.asset_durations.shape[1]}" for i in range(missing)]
                )
                self.asset_durations = pd.concat([self.asset_durations, synthetic], axis=1)
        
        if self.asset_convexities.shape[1] != expected_cols:
            logger.warning(f"Asset convexities shape mismatch. Expected {expected_cols} columns, got {self.asset_convexities.shape[1]}.")
            # Pad with synthetic data if needed
            if self.asset_convexities.shape[1] < expected_cols:
                missing = expected_cols - self.asset_convexities.shape[1]
                synthetic = pd.DataFrame(
                    np.random.uniform(0.1, 1.0, size=(len(self.bond_prices), missing)),
                    index=self.bond_prices.index,
                    columns=[f"convexity_{i+self.asset_convexities.shape[1]}" for i in range(missing)]
                )
                self.asset_convexities = pd.concat([self.asset_convexities, synthetic], axis=1)
        
        # Calculate returns
        self.asset_returns = self.asset_prices.pct_change().fillna(0)
        
        logger.info(f"Created asset features with shapes: prices {self.asset_prices.shape}, durations {self.asset_durations.shape}, convexities {self.asset_convexities.shape}")
    
    def _define_observation_space(self):
        """
        Define the observation space.
        
        The observation space includes:
        1. Market features (interest rates, credit spreads, economic indicators)
        2. Portfolio characteristics (weights, duration, etc.)
        3. Past returns
        """
        # Count features
        # Market features breakdown:
        # - short_rate: 1
        # - slope/2s10s_spread: 1
        # - credit spreads (AAA, BBB, B): 3
        # - regime one-hot encoding: 4
        # - extra feature for fixed dimension: 1
        num_market_features = 10  # 1 + 1 + 3 + 4 + 1 = 10
        
        # Portfolio features: weights, duration, convexity
        num_portfolio_features = 3 * self.num_assets
        
        # Past returns
        num_past_returns = self.window_size * self.num_assets
        
        # Total features
        total_features = num_market_features + num_portfolio_features + num_past_returns
        
        # Log feature dimensions for debugging
        logger.info(f"Defining observation space with dimensions:")
        logger.info(f"  Market features: {num_market_features}")
        logger.info(f"  Portfolio features: {num_portfolio_features} (3 * {self.num_assets} assets)")
        logger.info(f"  Past returns: {num_past_returns} ({self.window_size} window * {self.num_assets} assets)")
        logger.info(f"  Total: {total_features}")
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_features,), dtype=np.float32
        )
    
    def reset(self, *, seed=None, options=None):
        """
        Reset the environment.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Initial observation and info
        """
        # Initialize RNG
        super().reset(seed=seed)
        
        # Determine starting point
        if self.random_start:
            self.current_step = self.np_random.integers(self.window_size, self.max_steps - 1)
        else:
            self.current_step = self.window_size
        
        # Initialize portfolio
        self.initialize_portfolio()
        
        # Get initial observation
        observation = self._get_observation()
        
        # Debug observation dimensions
        self._debug_observation_dimensions(observation)
        
        info = {}
        
        return observation, info
    
    def initialize_portfolio(self):
        """
        Initialize the portfolio with equal weights.
        """
        # Initialize portfolio values
        self.portfolio_value = self.initial_cash
        self.portfolio_weights = np.ones(self.num_assets) / self.num_assets
        self.portfolio_holdings = self.portfolio_value * self.portfolio_weights
        
        # Initialize portfolio history
        self.portfolio_history = {
            'value': [self.portfolio_value],
            'weights': [self.portfolio_weights.copy()],
            'returns': [0.0],
            'duration': [self._calculate_portfolio_duration()],
            'convexity': [self._calculate_portfolio_convexity()],
            'regime': [self._get_current_regime()],
            'timestamp': [self.rates_data.index[self.current_step]]
        }
        
        # Initialize performance metrics
        self.cumulative_return = 0.0
        self.daily_returns = []
        self.turnover_history = []
        self.excess_returns = []
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0
        
        # Initialize rebalance counter
        self.days_since_rebalance = 0
    
    def _calculate_portfolio_duration(self):
        """
        Calculate the duration of the portfolio.
        
        Returns:
            Portfolio duration
        """
        # Get current durations
        current_durations = self.asset_durations.iloc[self.current_step].values
        
        # Calculate weighted duration
        portfolio_duration = np.sum(self.portfolio_weights * current_durations)
        
        return portfolio_duration
    
    def _calculate_portfolio_convexity(self):
        """
        Calculate the convexity of the portfolio.
        
        Returns:
            Portfolio convexity
        """
        # Get current convexities
        current_convexities = self.asset_convexities.iloc[self.current_step].values
        
        # Calculate weighted convexity
        portfolio_convexity = np.sum(self.portfolio_weights * current_convexities)
        
        return portfolio_convexity
    
    def _get_current_regime(self):
        """
        Get the current market regime.
        
        Returns:
            Current regime
        """
        if 'regime' in self.rates_data.columns:
            return self.rates_data['regime'].iloc[self.current_step]
        else:
            return 0  # Default regime if not specified
    
    def _get_observation(self):
        """
        Get the current observation.
        
        Returns:
            Observation array
        """
        # 1. Market features
        market_features = []
        
        # Interest rates
        short_rate = self.rates_data['short_rate'].iloc[self.current_step]
        market_features.append(short_rate)
        
        # Yield curve slope (2s10s spread)
        if 'slope' in self.rates_data.columns:
            slope = self.rates_data['slope'].iloc[self.current_step]
            market_features.append(slope)
        elif '2s10s_spread' in self.rates_data.columns:
            slope = self.rates_data['2s10s_spread'].iloc[self.current_step]
            market_features.append(slope)
        else:
            market_features.append(0.0)  # Default if not available
        
        # Credit spreads
        for rating in ['AAA', 'BBB', 'B']:
            col = f"spread_{rating}"
            if col in self.rates_data.columns:
                spread = self.rates_data[col].iloc[self.current_step]
                market_features.append(spread)
            else:
                market_features.append(0.0)  # Default if not available
        
        # Regime
        if self.regime_aware and 'regime' in self.rates_data.columns:
            # One-hot encode regime
            regime = self.rates_data['regime'].iloc[self.current_step]
            # Use fixed 4 regimes as defined in _define_observation_space
            regime_one_hot = np.zeros(4)
            # Ensure regime index is within bounds
            regime_index = min(int(regime), 3)  # Limit to 0-3 range
            regime_one_hot[regime_index] = 1
            market_features.extend(regime_one_hot)
        else:
            # Add placeholder zeros
            market_features.extend([0.0] * 4)  # Default 4 regimes
        
        # Extra market feature to ensure exactly 10 features
        # This matches the observation space definition
        market_features.append(0.0)
        
        # 2. Portfolio characteristics
        portfolio_features = []
        
        # Portfolio weights
        portfolio_features.extend(self.portfolio_weights)
        
        # Asset durations
        current_durations = self.asset_durations.iloc[self.current_step].values
        portfolio_features.extend(current_durations)
        
        # Asset convexities
        current_convexities = self.asset_convexities.iloc[self.current_step].values
        portfolio_features.extend(current_convexities)
        
        # 3. Past returns
        past_returns = []
        
        # Get window of past returns for each asset
        for i in range(self.num_assets):
            asset_returns = self.asset_returns.iloc[self.current_step - self.window_size:self.current_step][f"price_{i}"].values
            past_returns.extend(asset_returns)
        
        # Convert lists to numpy arrays for detailed debugging
        market_features = np.array(market_features, dtype=np.float32)
        portfolio_features = np.array(portfolio_features, dtype=np.float32)
        past_returns = np.array(past_returns, dtype=np.float32)
        
        # Log detailed feature counts for debugging
        logger.debug(f"Market features: {len(market_features)}, expected 10")
        logger.debug(f"Portfolio features: {len(portfolio_features)}, expected {3 * self.num_assets}")
        logger.debug(f"Past returns: {len(past_returns)}, expected {self.window_size * self.num_assets}")
        
        # Check if any asset has missing past returns
        if len(past_returns) != self.window_size * self.num_assets:
            for i in range(self.num_assets):
                asset_returns_len = len(self.asset_returns.iloc[self.current_step - self.window_size:self.current_step][f"price_{i}"].values)
                logger.debug(f"Asset {i} past returns length: {asset_returns_len}, expected {self.window_size}")
        
        # Combine features
        observation = np.concatenate([
            market_features,
            portfolio_features,
            past_returns
        ])
        
        # Add an extra zero if we're consistently missing one element
        # This is a temporary fix until we identify the exact issue
        expected_shape = self.observation_space.shape[0]
        actual_shape = observation.shape[0]
        
        if actual_shape == 669 and expected_shape == 670:
            # Add a padding value at the end for this specific case
            observation = np.append(observation, 0.0)
        elif actual_shape < expected_shape:
            # General padding case
            padding = np.zeros(expected_shape - actual_shape, dtype=np.float32)
            observation = np.concatenate([observation, padding])
            logger.warning(f"Observation was smaller than expected ({actual_shape} vs {expected_shape}). Padded with zeros.")
        elif actual_shape > expected_shape:
            # Truncate if observation is larger than expected
            observation = observation[:expected_shape]
            logger.warning(f"Observation was larger than expected ({actual_shape} vs {expected_shape}). Truncated to match.")
        
        return observation
    
    def _debug_observation_dimensions(self, observation):
        """
        Debug helper to analyze observation dimensions.
        
        Args:
            observation: Current observation
            
        Returns:
            None, just logs details
        """
        # Calculate expected dimensions
        num_market_features = 10
        num_portfolio_features = 3 * self.num_assets
        num_past_returns = self.window_size * self.num_assets
        total_expected = num_market_features + num_portfolio_features + num_past_returns
        
        # Get actual dimension
        actual_dim = observation.shape[0]
        
        if actual_dim != total_expected:
            logger.warning(f"Observation dimension mismatch:")
            logger.warning(f"  Expected total: {total_expected}, Actual: {actual_dim}")
            logger.warning(f"  Expected market features: {num_market_features}")
            logger.warning(f"  Expected portfolio features: {num_portfolio_features} (3 * {self.num_assets} assets)")
            logger.warning(f"  Expected past returns: {num_past_returns} ({self.window_size} days * {self.num_assets} assets)")
            
            # Check if past returns could be the issue
            for i in range(self.num_assets):
                try:
                    data_range = self.asset_returns.iloc[self.current_step - self.window_size:self.current_step][f"price_{i}"]
                    if len(data_range) != self.window_size:
                        logger.warning(f"  Asset {i} past returns has {len(data_range)} values, expected {self.window_size}")
                except Exception as e:
                    logger.warning(f"  Error checking asset {i} past returns: {e}")
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (target portfolio weights)
            
        Returns:
            Observation, reward, terminated, truncated, info
        """
        # Normalize action to ensure it sums to 1
        action = np.array(action, dtype=np.float32)
        action = np.clip(action, 0, 1)
        action = action / np.sum(action)
        
        # Increment step counter
        prev_step = self.current_step
        self.current_step += 1
        
        # Increment days since rebalance
        self.days_since_rebalance += 1
        
        # Get current asset prices
        prev_prices = self.asset_prices.iloc[prev_step].values
        current_prices = self.asset_prices.iloc[self.current_step].values
        
        # Calculate asset returns
        asset_returns = current_prices / prev_prices - 1
        
        # Update portfolio value (mark-to-market)
        self.portfolio_holdings = self.portfolio_holdings * (1 + asset_returns)
        new_portfolio_value = np.sum(self.portfolio_holdings)
        
        # Calculate portfolio return
        if self.portfolio_value > 0:
            portfolio_return = (new_portfolio_value / self.portfolio_value) - 1
        else:
            logger.warning("Portfolio value is zero or negative, using 0.0 as portfolio return")
            portfolio_return = 0.0

        # Validate return
        if np.isnan(portfolio_return) or np.isinf(portfolio_return):
            logger.warning(f"Invalid portfolio return calculated: {portfolio_return}. Using 0.0 instead.")
            portfolio_return = 0.0
        
        self.daily_returns.append(portfolio_return)
        self.portfolio_value = new_portfolio_value
        
        # Calculate excess return
        excess_return = portfolio_return - self.risk_free_rate
        self.excess_returns.append(excess_return)
        
        # Update portfolio weights (due to price changes)
        self.portfolio_weights = self.portfolio_holdings / self.portfolio_value
        
        # Rebalance if necessary
        if self.days_since_rebalance >= self.rebalance_freq:
            # Calculate turnover
            turnover = np.sum(np.abs(self.portfolio_weights - action))
            
            # Apply transaction costs
            transaction_cost = turnover * self.transaction_cost
            self.portfolio_value *= (1 - transaction_cost)
            
            # Update weights and holdings
            self.portfolio_weights = action
            self.portfolio_holdings = self.portfolio_value * self.portfolio_weights
            
            # Reset rebalance counter
            self.days_since_rebalance = 0
            
            # Record turnover
            self.turnover_history.append(turnover)
        else:
            # Record zero turnover (no rebalancing)
            self.turnover_history.append(0.0)
        
        # Calculate portfolio duration and convexity
        portfolio_duration = self._calculate_portfolio_duration()
        portfolio_convexity = self._calculate_portfolio_convexity()
        
        # Get current regime
        current_regime = self._get_current_regime()
        
        # Update portfolio history
        self.portfolio_history['value'].append(self.portfolio_value)
        self.portfolio_history['weights'].append(self.portfolio_weights.copy())
        self.portfolio_history['returns'].append(portfolio_return)
        self.portfolio_history['duration'].append(portfolio_duration)
        self.portfolio_history['convexity'].append(portfolio_convexity)
        self.portfolio_history['regime'].append(current_regime)
        self.portfolio_history['timestamp'].append(self.rates_data.index[self.current_step])
        
        # Calculate cumulative return
        self.cumulative_return = self.portfolio_value / self.initial_cash - 1
        
        # Calculate drawdown
        peak = np.maximum.accumulate(self.portfolio_history['value'])
        drawdown = (peak - self.portfolio_history['value']) / peak
        self.max_drawdown = max(drawdown)
        
        # Check for duration constraint violation
        duration_violation = 0.0
        if self.duration_constraint is not None:
            min_duration, max_duration = self.duration_constraint
            if portfolio_duration < min_duration:
                duration_violation = min_duration - portfolio_duration
            elif portfolio_duration > max_duration:
                duration_violation = portfolio_duration - max_duration
        
        # Calculate reward
        reward = self._calculate_reward(portfolio_return, duration_violation)
        
        # Check if done
        terminated = self.current_step >= self.max_steps - 1
        truncated = False
        
        # Get observation
        observation = self._get_observation()
        
        # Debug observation dimensions
        self._debug_observation_dimensions(observation)
        
        # Create info dictionary
        info = {
            'portfolio_value': self.portfolio_value,
            'portfolio_weights': self.portfolio_weights,
            'portfolio_return': portfolio_return,
            'portfolio_duration': portfolio_duration,
            'portfolio_convexity': portfolio_convexity,
            'cumulative_return': self.cumulative_return,
            'max_drawdown': self.max_drawdown,
            'regime': current_regime,
            'duration_violation': duration_violation
        }
        
        # Calculate additional metrics if episode is done
        if terminated:
            # Calculate Sharpe ratio
            if len(self.excess_returns) > 1:
                self.sharpe_ratio = np.mean(self.excess_returns) / (np.std(self.excess_returns) + 1e-6) * np.sqrt(252)
                info['sharpe_ratio'] = self.sharpe_ratio
            
            # Calculate volatility
            if len(self.daily_returns) > 1:
                volatility = np.std(self.daily_returns) * np.sqrt(252)
                info['volatility'] = volatility
            
            # Calculate average turnover
            if len(self.turnover_history) > 0:
                avg_turnover = np.mean(self.turnover_history)
                info['avg_turnover'] = avg_turnover
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(self, portfolio_return: float, duration_violation: float) -> float:
        """
        Calculate the reward.
        
        Args:
            portfolio_return: Portfolio return
            duration_violation: Duration constraint violation
            
        Returns:
            Reward value
        """
        # Check for invalid portfolio return
        if np.isnan(portfolio_return) or np.isinf(portfolio_return):
            logger.warning(f"Invalid portfolio return detected: {portfolio_return}. Using 0.0 instead.")
            portfolio_return = 0.0
        
        # Use the reward function if available
        if hasattr(self, 'reward_function'):
            # Pass additional information to the reward function
            reward = self.reward_function.calculate_reward(
                portfolio_return=portfolio_return,
                duration_violation=duration_violation,
                portfolio_duration=self._calculate_portfolio_duration(),
                target_duration=np.mean(self.duration_constraint) if self.duration_constraint else None,
                regime=self._get_current_regime() if self.regime_aware else None,
                returns_history=self.daily_returns,
                turnover=self.turnover_history[-1] if self.turnover_history else 0.0,
            )
            
            # Check for NaN reward
            if np.isnan(reward) or np.isinf(reward):
                logger.warning(f"NaN or Inf reward detected from reward function. Using fallback reward.")
                # Use fallback reward calculation with new scaled approach
                reward = 0.0  # Base reward
                reward += portfolio_return * 50.0  # Add immediate return component
                reward -= np.sqrt(duration_violation) * 0.2 if duration_violation > 0 else 0.0
                # Clip to reasonable range
                reward = max(min(reward, 1.0), -1.0)
            
            return reward
        
        # Fallback to the original reward calculation, but with new scaling
        # Start with base reward
        reward = 0.0
        
        # Add immediate return component
        immediate_return = portfolio_return * 50.0
        immediate_return = max(min(immediate_return, 0.5), -0.5)  # Cap impact
        reward += immediate_return
        
        # Calculate rolling Sharpe for risk_adjusted and sharpe reward types
        if self.reward_type == 'sharpe' or self.reward_type == 'risk_adjusted':
            if len(self.daily_returns) >= 3:
                # Filter out invalid values
                valid_returns = [r for r in self.daily_returns[-10:] if not (np.isnan(r) or np.isinf(r))]
                if len(valid_returns) >= 3:
                    # Calculate either Sharpe or risk-adjusted component
                    if self.reward_type == 'sharpe':
                        excess_returns = [r - self.risk_free_rate for r in valid_returns]
                        mean_excess = np.mean(excess_returns)
                        std_excess = max(np.std(excess_returns), 1e-5)
                        ratio = mean_excess / std_excess * np.sqrt(252)
                        # Use tanh for better scaling
                        reward_component = np.tanh(ratio)
                    else:  # risk_adjusted
                        std_returns = max(np.std(valid_returns), 1e-5)
                        ratio = portfolio_return / std_returns
                        # Use tanh for better scaling
                        reward_component = np.tanh(ratio * 10.0)
                    
                    # Add component with appropriate weight
                    reward += reward_component * 0.5
        
        # Apply penalty for duration violation with reduced impact
        if duration_violation > 0:
            # Use square root to reduce impact of large violations
            duration_penalty = -np.sqrt(duration_violation) * 0.2
            reward += duration_penalty
        
        # Final safety check
        if np.isnan(reward) or np.isinf(reward):
            logger.warning(f"NaN or Inf reward detected after calculation. Using 0.0 as fallback.")
            reward = 0.0
        
        # Clip to reasonable range
        reward = max(min(reward, 1.0), -1.0)
        
        return reward
    
    def get_portfolio_history_df(self) -> pd.DataFrame:
        """
        Get the portfolio history as a DataFrame.
        
        Returns:
            DataFrame with portfolio history
        """
        # Convert to DataFrame
        df = pd.DataFrame({
            'value': self.portfolio_history['value'],
            'return': self.portfolio_history['returns'],
            'duration': self.portfolio_history['duration'],
            'convexity': self.portfolio_history['convexity'],
            'regime': self.portfolio_history['regime']
        }, index=self.portfolio_history['timestamp'])
        
        # Add weight columns
        for i in range(self.num_assets):
            df[f'weight_{i}'] = [weights[i] for weights in self.portfolio_history['weights']]
        
        return df
    
    def plot_portfolio_performance(self) -> plt.Figure:
        """
        Plot portfolio performance.
        
        Returns:
            Matplotlib figure
        """
        # Get portfolio history
        history_df = self.get_portfolio_history_df()
        
        # Create figure
        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Plot portfolio value
        axs[0].plot(history_df.index, history_df['value'], 'b-', linewidth=2)
        axs[0].set_ylabel('Portfolio Value')
        axs[0].set_title('Portfolio Performance')
        axs[0].grid(True)
        
        # Plot portfolio returns
        axs[1].plot(history_df.index, history_df['return'] * 100, 'g-', linewidth=1)
        axs[1].set_ylabel('Daily Return (%)')
        axs[1].grid(True)
        
        # Plot portfolio duration
        axs[2].plot(history_df.index, history_df['duration'], 'r-', linewidth=2)
        axs[2].set_ylabel('Duration')
        axs[2].set_xlabel('Date')
        axs[2].grid(True)
        
        # Add horizontal lines for duration constraints
        if self.duration_constraint is not None:
            min_duration, max_duration = self.duration_constraint
            axs[2].axhline(min_duration, color='k', linestyle='--', linewidth=1)
            axs[2].axhline(max_duration, color='k', linestyle='--', linewidth=1)
        
        # Layout
        plt.tight_layout()
        
        return fig
    
    def plot_weights_evolution(self) -> plt.Figure:
        """
        Plot portfolio weights evolution.
        
        Returns:
            Matplotlib figure
        """
        # Get portfolio history
        history_df = self.get_portfolio_history_df()
        
        # Get weight columns
        weight_cols = [col for col in history_df.columns if 'weight_' in col]
        
        # Rename weight columns to use asset names if available
        if hasattr(self, 'selected_assets'):
            weight_cols_map = {f'weight_{i}': self.selected_assets[i] for i in range(self.num_assets)}
            history_df = history_df.rename(columns=weight_cols_map)
            weight_cols = [weight_cols_map[col] for col in weight_cols]
        
        # Create figure
        fig, axs = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot stacked weights
        history_df[weight_cols].plot(kind='area', stacked=True, ax=axs[0])
        axs[0].set_ylabel('Weight')
        axs[0].set_title('Portfolio Weights Evolution')
        axs[0].grid(True)
        axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
        
        # Plot regimes
        if 'regime' in history_df.columns:
            # Create a colormap for regimes
            unique_regimes = history_df['regime'].unique()
            cmap = plt.cm.get_cmap('tab10', len(unique_regimes))
            
            # Create a scatter plot for regimes
            for regime in unique_regimes:
                regime_periods = history_df[history_df['regime'] == regime]
                axs[1].scatter(
                    regime_periods.index, 
                    np.ones(len(regime_periods)) * regime,
                    c=[cmap(regime)], 
                    s=50, 
                    label=f'Regime {regime}'
                )
            
            axs[1].set_ylabel('Regime')
            axs[1].set_title('Market Regimes')
            axs[1].grid(True)
            axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
        
        # Layout
        plt.tight_layout()
        
        return fig


class FixedIncomeDurationEnv(FixedIncomeEnv):
    """
    Variant of FixedIncomeEnv where actions represent duration targets instead of portfolio weights.
    
    This allows for higher-level portfolio management strategies based on duration targets.
    """
    
    def __init__(self, 
                 market_data: Dict[str, pd.DataFrame],
                 bond_universe: pd.DataFrame,
                 num_assets: int,
                 initial_cash: float = 1000000.0,
                 rebalance_freq: int = 21,
                 transaction_cost: float = 0.0005,
                 reward_type: str = 'sharpe',
                 reward_function: Optional[RewardFunction] = None,
                 regime_aware: bool = True,
                 duration_constraint: Optional[Tuple[float, float]] = (1.0, 10.0),
                 random_start: bool = True,
                 window_size: int = 63,
                 risk_free_rate: float = 0.0):
        """
        Initialize the fixed income duration-based environment.
        
        Args:
            Same as FixedIncomeEnv, but duration_constraint is required for duration targeting.
        """
        super(FixedIncomeDurationEnv, self).__init__(
            market_data=market_data,
            bond_universe=bond_universe,
            num_assets=num_assets,
            initial_cash=initial_cash,
            rebalance_freq=rebalance_freq,
            transaction_cost=transaction_cost,
            reward_type=reward_type,
            reward_function=reward_function,
            regime_aware=regime_aware,
            duration_constraint=duration_constraint,
            random_start=random_start,
            window_size=window_size,
            risk_free_rate=risk_free_rate
        )
        
        # Override action space for duration targeting
        # Action is a target duration in range [min_duration, max_duration]
        if duration_constraint is not None:
            min_duration, max_duration = duration_constraint
            self.action_space = spaces.Box(
                low=min_duration, high=max_duration, shape=(1,), dtype=np.float32
            )
        else:
            self.action_space = spaces.Box(
                low=0.0, high=10.0, shape=(1,), dtype=np.float32
            )
        
        # Import portfolio optimizer
        from src.models.portfolio_optimizer import FixedIncomePortfolioOptimizer
        self.optimizer = FixedIncomePortfolioOptimizer(risk_aversion=1.0, transaction_cost=transaction_cost)
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (target duration)
            
        Returns:
            Observation, reward, terminated, truncated, info
        """
        # Get target duration from action
        target_duration = float(action[0])
        
        # Increment step counter
        prev_step = self.current_step
        self.current_step += 1
        
        # Increment days since rebalance
        self.days_since_rebalance += 1
        
        # Get current asset prices
        prev_prices = self.asset_prices.iloc[prev_step].values
        current_prices = self.asset_prices.iloc[self.current_step].values
        
        # Calculate asset returns
        asset_returns = current_prices / prev_prices - 1
        
        # Update portfolio value (mark-to-market)
        self.portfolio_holdings = self.portfolio_holdings * (1 + asset_returns)
        new_portfolio_value = np.sum(self.portfolio_holdings)
        
        # Calculate portfolio return
        if self.portfolio_value > 0:
            portfolio_return = (new_portfolio_value / self.portfolio_value) - 1
        else:
            logger.warning("Portfolio value is zero or negative, using 0.0 as portfolio return")
            portfolio_return = 0.0

        # Validate return
        if np.isnan(portfolio_return) or np.isinf(portfolio_return):
            logger.warning(f"Invalid portfolio return calculated: {portfolio_return}. Using 0.0 instead.")
            portfolio_return = 0.0
        
        self.daily_returns.append(portfolio_return)
        self.portfolio_value = new_portfolio_value
        
        # Calculate excess return
        excess_return = portfolio_return - self.risk_free_rate
        self.excess_returns.append(excess_return)
        
        # Update portfolio weights (due to price changes)
        self.portfolio_weights = self.portfolio_holdings / self.portfolio_value
        
        # Rebalance if necessary
        if self.days_since_rebalance >= self.rebalance_freq:
            # Get current durations
            current_durations = self.asset_durations.iloc[self.current_step].values
            
            # Get expected returns (using historical returns for simplicity)
            recent_returns = self.asset_returns.iloc[max(0, self.current_step - 30):self.current_step].mean().values
            
            # Get covariance matrix (using historical returns for simplicity)
            cov_matrix = self.asset_returns.iloc[max(0, self.current_step - 30):self.current_step].cov().values
            
            # Optimize portfolio to match target duration
            try:
                optimal_weights = self.optimizer.duration_matching_optimization(
                    expected_returns=recent_returns,
                    covariance_matrix=cov_matrix,
                    durations=current_durations,
                    target_duration=target_duration,
                    current_weights=self.portfolio_weights,
                    constraints={'long_only': True}
                )
            except Exception as e:
                # If optimization fails, keep current weights
                logger.warning(f"Portfolio optimization failed: {e}")
                optimal_weights = self.portfolio_weights
            
            if optimal_weights is None:
                # If optimization fails, keep current weights
                optimal_weights = self.portfolio_weights
            
            # Calculate turnover
            turnover = np.sum(np.abs(self.portfolio_weights - optimal_weights))
            
            # Apply transaction costs
            transaction_cost = turnover * self.transaction_cost
            self.portfolio_value *= (1 - transaction_cost)
            
            # Update weights and holdings
            self.portfolio_weights = optimal_weights
            self.portfolio_holdings = self.portfolio_value * self.portfolio_weights
            
            # Reset rebalance counter
            self.days_since_rebalance = 0
            
            # Record turnover
            self.turnover_history.append(turnover)
        else:
            # Record zero turnover (no rebalancing)
            self.turnover_history.append(0.0)
        
        # Calculate portfolio duration and convexity
        portfolio_duration = self._calculate_portfolio_duration()
        portfolio_convexity = self._calculate_portfolio_convexity()
        
        # Get current regime
        current_regime = self._get_current_regime()
        
        # Update portfolio history
        self.portfolio_history['value'].append(self.portfolio_value)
        self.portfolio_history['weights'].append(self.portfolio_weights.copy())
        self.portfolio_history['returns'].append(portfolio_return)
        self.portfolio_history['duration'].append(portfolio_duration)
        self.portfolio_history['convexity'].append(portfolio_convexity)
        self.portfolio_history['regime'].append(current_regime)
        self.portfolio_history['timestamp'].append(self.rates_data.index[self.current_step])
        
        # Calculate cumulative return
        self.cumulative_return = self.portfolio_value / self.initial_cash - 1
        
        # Calculate drawdown
        peak = np.maximum.accumulate(self.portfolio_history['value'])
        drawdown = (peak - self.portfolio_history['value']) / peak
        self.max_drawdown = max(drawdown)
        
        # Calculate duration error
        duration_error = abs(portfolio_duration - target_duration)
        
        # Check for duration constraint violation
        duration_violation = 0.0
        if self.duration_constraint is not None:
            min_duration, max_duration = self.duration_constraint
            if portfolio_duration < min_duration:
                duration_violation = min_duration - portfolio_duration
            elif portfolio_duration > max_duration:
                duration_violation = portfolio_duration - max_duration
        
        # Calculate reward
        reward = self._calculate_reward(portfolio_return, duration_violation)
        
        # Apply additional penalty for duration error (with new scaling)
        if not np.isnan(duration_error) and not np.isinf(duration_error):
            # Use square root to reduce impact of large errors
            duration_error_penalty = -np.sqrt(duration_error) * 0.1
            reward += duration_error_penalty
            
            # Safety check for the reward
            if np.isnan(reward) or np.isinf(reward):
                logger.warning(f"NaN or Inf reward detected after duration error penalty. Using 0.0 as fallback.")
                reward = 0.0
            
            # Clip to reasonable range
            reward = max(min(reward, 1.0), -1.0)
        
        # Check if done
        terminated = self.current_step >= self.max_steps - 1
        truncated = False
        
        # Get observation
        observation = self._get_observation()
        
        # Debug observation dimensions
        self._debug_observation_dimensions(observation)
        
        # Create info dictionary
        info = {
            'portfolio_value': self.portfolio_value,
            'portfolio_weights': self.portfolio_weights,
            'portfolio_return': portfolio_return,
            'portfolio_duration': portfolio_duration,
            'target_duration': target_duration,
            'duration_error': duration_error,
            'portfolio_convexity': portfolio_convexity,
            'cumulative_return': self.cumulative_return,
            'max_drawdown': self.max_drawdown,
            'regime': current_regime,
            'duration_violation': duration_violation
        }
        
        # Calculate additional metrics if episode is done
        if terminated:
            # Calculate Sharpe ratio
            if len(self.excess_returns) > 1:
                self.sharpe_ratio = np.mean(self.excess_returns) / (np.std(self.excess_returns) + 1e-6) * np.sqrt(252)
                info['sharpe_ratio'] = self.sharpe_ratio
            
            # Calculate volatility
            if len(self.daily_returns) > 1:
                volatility = np.std(self.daily_returns) * np.sqrt(252)
                info['volatility'] = volatility
            
            # Calculate average turnover
            if len(self.turnover_history) > 0:
                avg_turnover = np.mean(self.turnover_history)
                info['avg_turnover'] = avg_turnover
        
        return observation, reward, terminated, truncated, info
