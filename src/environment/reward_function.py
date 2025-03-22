"""
Reward Function Module for Fixed Income RL Project

This module implements various reward functions for reinforcement learning
in fixed income portfolio management:

1. Return-based rewards (simple, excess, risk-adjusted)
2. Sharpe ratio-based rewards
3. Duration targeting rewards
4. Multi-objective rewards with constraints

Mathematical foundations:
- Risk-adjusted returns
- Regime-dependent benchmarks
- Duration and convexity penalties
- Transaction cost penalties

Author: ranycs & cosrv
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RewardFunction:
    """
    Base class for reward functions.
    """
    
    def __init__(self, risk_free_rate: float = 0.0):
        """
        Initialize the reward function.
        
        Args:
            risk_free_rate: Risk-free rate for calculating excess returns
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_reward(self, portfolio_return: float, **kwargs) -> float:
        """
        Calculate the reward.
        
        Args:
            portfolio_return: Portfolio return
            **kwargs: Additional keyword arguments
            
        Returns:
            Reward value
        """
        raise NotImplementedError("Subclasses must implement this method")


class ReturnReward(RewardFunction):
    """
    Reward function based on raw returns.
    """
    
    def __init__(self, scaling_factor: float = 100.0, risk_free_rate: float = 0.0):
        """
        Initialize the return reward function.
        
        Args:
            scaling_factor: Factor to scale returns for better learning
            risk_free_rate: Risk-free rate for calculating excess returns
        """
        super(ReturnReward, self).__init__(risk_free_rate=risk_free_rate)
        self.scaling_factor = scaling_factor
    
    def calculate_reward(self, portfolio_return: float, **kwargs) -> float:
        """
        Calculate the reward based on raw returns.
        
        Args:
            portfolio_return: Portfolio return
            **kwargs: Additional keyword arguments
            
        Returns:
            Reward value
        """
        # Scale the return for better learning
        reward = portfolio_return * self.scaling_factor
        
        return reward


class ExcessReturnReward(RewardFunction):
    """
    Reward function based on excess returns over the risk-free rate.
    """
    
    def __init__(self, scaling_factor: float = 100.0, risk_free_rate: float = 0.0):
        """
        Initialize the excess return reward function.
        
        Args:
            scaling_factor: Factor to scale returns for better learning
            risk_free_rate: Risk-free rate for calculating excess returns
        """
        super(ExcessReturnReward, self).__init__(risk_free_rate=risk_free_rate)
        self.scaling_factor = scaling_factor
    
    def calculate_reward(self, portfolio_return: float, **kwargs) -> float:
        """
        Calculate the reward based on excess returns.
        
        Args:
            portfolio_return: Portfolio return
            **kwargs: Additional keyword arguments
            
        Returns:
            Reward value
        """
        # Calculate excess return
        excess_return = portfolio_return - self.risk_free_rate
        
        # Scale the excess return for better learning
        reward = excess_return * self.scaling_factor
        
        return reward


class SharpeReward(RewardFunction):
    """
    Reward function based on Sharpe ratio approximation.
    """
    
    def __init__(self, 
                window_size: int = 10, 
                annualization_factor: float = 252.0,
                scaling_factor: float = 1.0,
                base_reward: float = 0.0,
                min_reward: float = -1.0,
                max_reward: float = 1.0,
                min_std: float = 1e-5,
                penalty_factor: float = 0.2,
                risk_free_rate: float = 0.0):
        """
        Initialize the Sharpe ratio reward function.
        
        Args:
            window_size: Window size for calculating Sharpe ratio
            annualization_factor: Factor to annualize Sharpe ratio
            scaling_factor: Factor to scale Sharpe ratio for better learning
            base_reward: Base reward given for taking actions (positive bias)
            min_reward: Minimum reward value to prevent extreme negatives
            max_reward: Maximum reward value to prevent extreme positives
            min_std: Minimum standard deviation to avoid division by zero
            penalty_factor: Factor to scale penalties
            risk_free_rate: Risk-free rate for calculating excess returns
        """
        super(SharpeReward, self).__init__(risk_free_rate=risk_free_rate)
        self.window_size = window_size
        self.annualization_factor = annualization_factor
        self.scaling_factor = scaling_factor
        self.base_reward = base_reward
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.min_std = min_std
        self.penalty_factor = penalty_factor
        
        # Initialize return history
        self.return_history = []
    
    def calculate_reward(self, 
                       portfolio_return: float, 
                       duration_violation: float = 0.0,
                       **kwargs) -> float:
        """
        Calculate the reward based on Sharpe ratio approximation.
        
        Args:
            portfolio_return: Portfolio return
            duration_violation: Duration constraint violation
            **kwargs: Additional keyword arguments
            
        Returns:
            Reward value
        """
        # Ensure portfolio return is a valid number
        if np.isnan(portfolio_return) or np.isinf(portfolio_return):
            # Fallback to minimum reward if return is invalid
            return self.min_reward
            
        # Add current return to history
        self.return_history.append(portfolio_return)
        
        # Keep only the most recent returns in the window
        if len(self.return_history) > self.window_size:
            self.return_history = self.return_history[-self.window_size:]
        
        # Start with base reward (creates positive bias)
        reward = self.base_reward
        
        # Add immediate return component (makes learning more immediate)
        # Scale returns to be significant but not dominate
        immediate_return = portfolio_return * 50.0
        immediate_return = max(min(immediate_return, 0.5), -0.5)  # Cap impact
        reward += immediate_return
        
        # Calculate Sharpe ratio if we have enough data
        if len(self.return_history) >= 3:  # Need at least a few points for meaningful Sharpe
            # Filter out any NaN or inf values for safety
            valid_returns = [r for r in self.return_history if not (np.isnan(r) or np.isinf(r))]
            
            # If we don't have enough valid returns, skip this component
            if len(valid_returns) >= 3:
                excess_returns = [r - self.risk_free_rate for r in valid_returns]
                mean_excess = np.mean(excess_returns)
                std_excess = max(np.std(excess_returns), self.min_std)  # Ensure minimum std
                
                # Calculate Sharpe ratio with safeguards
                sharpe = mean_excess / std_excess * np.sqrt(self.annualization_factor)
                
                # Scale Sharpe ratio - use a sigmoid-like function for better scaling
                sharpe_component = np.tanh(sharpe * self.scaling_factor)
                reward += sharpe_component * 0.5  # Make Sharpe component significant but not dominant
        
        # Apply penalty for duration violation (reduce penalty impact)
        if duration_violation > 0:
            # Apply a smaller penalty (use square root to reduce impact of large violations)
            duration_penalty = -np.sqrt(duration_violation) * self.penalty_factor
            reward += duration_penalty
        
        # Clip final reward to reasonable range
        reward = max(min(reward, self.max_reward), self.min_reward)
        
        return reward


class RiskAdjustedReward(RewardFunction):
    """
    Reward function based on risk-adjusted returns.
    """
    
    def __init__(self, 
                window_size: int = 10, 
                scaling_factor: float = 100.0, 
                risk_free_rate: float = 0.0):
        """
        Initialize the risk-adjusted reward function.
        
        Args:
            window_size: Window size for calculating volatility
            scaling_factor: Factor to scale returns for better learning
            risk_free_rate: Risk-free rate for calculating excess returns
        """
        super(RiskAdjustedReward, self).__init__(risk_free_rate=risk_free_rate)
        self.window_size = window_size
        self.scaling_factor = scaling_factor
        
        # Initialize return history
        self.return_history = []
    
    def calculate_reward(self, portfolio_return: float, **kwargs) -> float:
        """
        Calculate the reward based on risk-adjusted returns.
        
        Args:
            portfolio_return: Portfolio return
            **kwargs: Additional keyword arguments
            
        Returns:
            Reward value
        """
        # Add current return to history
        self.return_history.append(portfolio_return)
        
        # Keep only the most recent returns in the window
        if len(self.return_history) > self.window_size:
            self.return_history = self.return_history[-self.window_size:]
        
        # Calculate risk-adjusted return if we have enough data
        if len(self.return_history) >= 3:  # Need at least a few points for meaningful std
            std_returns = np.std(self.return_history) + 1e-6  # Avoid division by zero
            risk_adjusted_return = portfolio_return / std_returns
            
            # Scale the risk-adjusted return for better learning
            reward = risk_adjusted_return * self.scaling_factor
        else:
            # Not enough data points, use simple return
            reward = portfolio_return * self.scaling_factor
        
        return reward


class BenchmarkRelativeReward(RewardFunction):
    """
    Reward function based on performance relative to a benchmark.
    """
    
    def __init__(self, 
                scaling_factor: float = 100.0, 
                risk_free_rate: float = 0.0,
                information_ratio: bool = False,
                window_size: int = 10):
        """
        Initialize the benchmark-relative reward function.
        
        Args:
            scaling_factor: Factor to scale returns for better learning
            risk_free_rate: Risk-free rate for calculating excess returns
            information_ratio: Whether to use information ratio instead of simple outperformance
            window_size: Window size for calculating information ratio
        """
        super(BenchmarkRelativeReward, self).__init__(risk_free_rate=risk_free_rate)
        self.scaling_factor = scaling_factor
        self.information_ratio = information_ratio
        self.window_size = window_size
        
        # Initialize history
        self.active_return_history = []
    
    def calculate_reward(self, portfolio_return: float, benchmark_return: float, **kwargs) -> float:
        """
        Calculate the reward based on performance relative to a benchmark.
        
        Args:
            portfolio_return: Portfolio return
            benchmark_return: Benchmark return
            **kwargs: Additional keyword arguments
            
        Returns:
            Reward value
        """
        # Calculate active return
        active_return = portfolio_return - benchmark_return
        
        if self.information_ratio:
            # Add active return to history
            self.active_return_history.append(active_return)
            
            # Keep only the most recent returns in the window
            if len(self.active_return_history) > self.window_size:
                self.active_return_history = self.active_return_history[-self.window_size:]
            
            # Calculate information ratio if we have enough data
            if len(self.active_return_history) >= 3:  # Need at least a few points
                mean_active = np.mean(self.active_return_history)
                std_active = np.std(self.active_return_history) + 1e-6  # Avoid division by zero
                information_ratio = mean_active / std_active * np.sqrt(252)
                
                # Use information ratio as reward
                reward = information_ratio
            else:
                # Not enough data points, use simple active return
                reward = active_return * self.scaling_factor
        else:
            # Scale the active return for better learning
            reward = active_return * self.scaling_factor
        
        return reward


class DurationTargetingReward(RewardFunction):
    """
    Reward function for duration targeting.
    """
    
    def __init__(self, 
                return_weight: float = 0.5, 
                duration_weight: float = 0.5,
                scaling_factor: float = 100.0,
                risk_free_rate: float = 0.0):
        """
        Initialize the duration targeting reward function.
        
        Args:
            return_weight: Weight for the return component
            duration_weight: Weight for the duration component
            scaling_factor: Factor to scale returns for better learning
            risk_free_rate: Risk-free rate for calculating excess returns
        """
        super(DurationTargetingReward, self).__init__(risk_free_rate=risk_free_rate)
        self.return_weight = return_weight
        self.duration_weight = duration_weight
        self.scaling_factor = scaling_factor
    
    def calculate_reward(self, 
                       portfolio_return: float, 
                       portfolio_duration: float,
                       target_duration: float,
                       **kwargs) -> float:
        """
        Calculate the reward based on return and duration targeting.
        
        Args:
            portfolio_return: Portfolio return
            portfolio_duration: Portfolio duration
            target_duration: Target duration
            **kwargs: Additional keyword arguments
            
        Returns:
            Reward value
        """
        # Calculate return component
        return_component = portfolio_return * self.scaling_factor
        
        # Calculate duration error
        duration_error = abs(portfolio_duration - target_duration)
        
        # Convert duration error to a penalty (negative reward)
        # Normalize by typical duration range
        duration_penalty = -duration_error * 2.0
        
        # Combine components
        reward = (
            self.return_weight * return_component +
            self.duration_weight * duration_penalty
        )
        
        return reward


class RegimeAwareReward(RewardFunction):
    """
    Reward function that adapts to different market regimes.
    """
    
    def __init__(self, 
                regime_weights: Dict[int, Dict[str, float]],
                window_size: int = 10,
                scaling_factor: float = 100.0,
                risk_free_rate: float = 0.0):
        """
        Initialize the regime-aware reward function.
        
        Args:
            regime_weights: Dictionary mapping regime indices to component weights
                Example: {0: {'return': 0.7, 'duration': 0.3}, 1: {'return': 0.3, 'duration': 0.7}}
            window_size: Window size for calculating volatility
            scaling_factor: Factor to scale returns for better learning
            risk_free_rate: Risk-free rate for calculating excess returns
        """
        super(RegimeAwareReward, self).__init__(risk_free_rate=risk_free_rate)
        self.regime_weights = regime_weights
        self.window_size = window_size
        self.scaling_factor = scaling_factor
        
        # Initialize return history
        self.return_history = []
    
    def calculate_reward(self, 
                       portfolio_return: float, 
                       regime: int,
                       portfolio_duration: Optional[float] = None,
                       target_duration: Optional[float] = None,
                       **kwargs) -> float:
        """
        Calculate the reward based on the current regime.
        
        Args:
            portfolio_return: Portfolio return
            regime: Current market regime
            portfolio_duration: Portfolio duration (optional)
            target_duration: Target duration (optional)
            **kwargs: Additional keyword arguments
            
        Returns:
            Reward value
        """
        # Add current return to history
        self.return_history.append(portfolio_return)
        
        # Keep only the most recent returns in the window
        if len(self.return_history) > self.window_size:
            self.return_history = self.return_history[-self.window_size:]
        
        # Get weights for the current regime
        if regime in self.regime_weights:
            weights = self.regime_weights[regime]
        else:
            # Default weights if regime not found
            weights = {'return': 1.0}
            logger.warning(f"Regime {regime} not found in regime_weights. Using default weights.")
        
        # Calculate components
        components = {}
        
        # Return component
        if 'return' in weights:
            components['return'] = portfolio_return * self.scaling_factor
        
        # Excess return component
        if 'excess_return' in weights:
            components['excess_return'] = (portfolio_return - self.risk_free_rate) * self.scaling_factor
        
        # Risk-adjusted return component
        if 'risk_adjusted' in weights and len(self.return_history) >= 3:
            std_returns = np.std(self.return_history) + 1e-6  # Avoid division by zero
            components['risk_adjusted'] = (portfolio_return / std_returns) * self.scaling_factor
        
        # Sharpe ratio component
        if 'sharpe' in weights and len(self.return_history) >= 3:
            excess_returns = [r - self.risk_free_rate for r in self.return_history]
            mean_excess = np.mean(excess_returns)
            std_excess = np.std(excess_returns) + 1e-6  # Avoid division by zero
            components['sharpe'] = mean_excess / std_excess * np.sqrt(252)
        
        # Duration targeting component
        if 'duration' in weights and portfolio_duration is not None and target_duration is not None:
            duration_error = abs(portfolio_duration - target_duration)
            components['duration'] = -duration_error * 2.0
        
        # Combine components
        reward = 0.0
        for component, value in components.items():
            if component in weights:
                reward += weights[component] * value
        
        return reward


class MultiObjectiveReward(RewardFunction):
    """
    Reward function combining multiple objectives with constraints.
    """
    
    def __init__(self, 
                return_weight: float = 0.5,
                risk_weight: float = 0.3,
                duration_weight: float = 0.2,
                constraint_penalty: float = 2.0,
                window_size: int = 10,
                scaling_factor: float = 100.0,
                risk_free_rate: float = 0.0):
        """
        Initialize the multi-objective reward function.
        
        Args:
            return_weight: Weight for the return component
            risk_weight: Weight for the risk component
            duration_weight: Weight for the duration component
            constraint_penalty: Penalty factor for constraint violations
            window_size: Window size for calculating volatility
            scaling_factor: Factor to scale returns for better learning
            risk_free_rate: Risk-free rate for calculating excess returns
        """
        super(MultiObjectiveReward, self).__init__(risk_free_rate=risk_free_rate)
        self.return_weight = return_weight
        self.risk_weight = risk_weight
        self.duration_weight = duration_weight
        self.constraint_penalty = constraint_penalty
        self.window_size = window_size
        self.scaling_factor = scaling_factor
        
        # Initialize return history
        self.return_history = []
    
    def calculate_reward(self, 
                       portfolio_return: float,
                       duration_violation: float = 0.0,
                       turnover: float = 0.0,
                       max_turnover: Optional[float] = None,
                       portfolio_duration: Optional[float] = None,
                       target_duration: Optional[float] = None,
                       **kwargs) -> float:
        """
        Calculate the reward based on multiple objectives with constraints.
        
        Args:
            portfolio_return: Portfolio return
            duration_violation: Duration constraint violation
            turnover: Portfolio turnover
            max_turnover: Maximum allowed turnover
            portfolio_duration: Portfolio duration
            target_duration: Target duration
            **kwargs: Additional keyword arguments
            
        Returns:
            Reward value
        """
        # Add current return to history
        self.return_history.append(portfolio_return)
        
        # Keep only the most recent returns in the window
        if len(self.return_history) > self.window_size:
            self.return_history = self.return_history[-self.window_size:]
        
        # Calculate return component
        return_component = portfolio_return * self.scaling_factor
        
        # Calculate risk component
        if len(self.return_history) >= 3:
            volatility = np.std(self.return_history)
            risk_component = -volatility * self.scaling_factor  # Negative because lower risk is better
        else:
            risk_component = 0.0
        
        # Calculate duration component
        if portfolio_duration is not None and target_duration is not None:
            duration_error = abs(portfolio_duration - target_duration)
            duration_component = -duration_error * 2.0
        else:
            duration_component = 0.0
        
        # Calculate penalty for constraint violations
        penalty = 0.0
        
        # Duration constraint violation
        if duration_violation > 0:
            penalty += duration_violation * self.constraint_penalty
        
        # Turnover constraint violation
        if max_turnover is not None and turnover > max_turnover:
            penalty += (turnover - max_turnover) * self.constraint_penalty
        
        # Combine components
        reward = (
            self.return_weight * return_component +
            self.risk_weight * risk_component +
            self.duration_weight * duration_component -
            penalty
        )
        
        return reward


# Factory function to create reward functions
def create_reward_function(reward_type: str, **kwargs) -> RewardFunction:
    """
    Create a reward function of the specified type.
    
    Args:
        reward_type: Type of reward function
        **kwargs: Additional keyword arguments for the reward function
        
    Returns:
        RewardFunction instance
    """
    if reward_type == 'return':
        return ReturnReward(**kwargs)
    elif reward_type == 'excess_return':
        return ExcessReturnReward(**kwargs)
    elif reward_type == 'sharpe':
        # Set default values for SharpeReward if not provided
        if 'scaling_factor' not in kwargs:
            kwargs['scaling_factor'] = 1.0
        if 'min_reward' not in kwargs:
            kwargs['min_reward'] = -1.0
        if 'max_reward' not in kwargs:
            kwargs['max_reward'] = 1.0
        if 'min_std' not in kwargs:
            kwargs['min_std'] = 1e-5
        if 'penalty_factor' not in kwargs:
            kwargs['penalty_factor'] = 0.2
        return SharpeReward(**kwargs)
    elif reward_type == 'risk_adjusted':
        return RiskAdjustedReward(**kwargs)
    elif reward_type == 'benchmark':
        return BenchmarkRelativeReward(**kwargs)
    elif reward_type == 'duration':
        return DurationTargetingReward(**kwargs)
    elif reward_type == 'regime':
        return RegimeAwareReward(**kwargs)
    elif reward_type == 'multi':
        return MultiObjectiveReward(**kwargs)
    else:
        logger.warning(f"Unknown reward type '{reward_type}'. Using default ReturnReward.")
        return ReturnReward(**kwargs)
