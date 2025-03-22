"""
Portfolio Optimizer Module for Fixed Income RL Project

This module implements portfolio optimization techniques:
1. Mean-variance optimization
2. Risk-based portfolio construction
3. Duration-constrained optimization
4. Transaction cost optimization

Mathematical foundations:
- Convex optimization (CVXPY)
- Risk metrics (variance, VaR, CVaR)
- Portfolio constraints (duration, convexity, rating buckets)
- Transaction cost modeling

Author: ranycs & cosrv
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Dict, List, Tuple, Optional, Union
import logging
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FixedIncomePortfolioOptimizer:
    """
    Class for optimizing fixed income portfolios with constraints.
    """
    
    def __init__(self, risk_aversion: float = 1.0, transaction_cost: float = 0.0005):
        """
        Initialize the portfolio optimizer.
        
        Args:
            risk_aversion: Risk aversion parameter (higher means more risk-averse)
            transaction_cost: Transaction cost as a fraction of traded value
        """
        self.risk_aversion = risk_aversion
        self.transaction_cost = transaction_cost
        
        # Store optimization results
        self.optimal_weights = None
        self.expected_return = None
        self.expected_risk = None
        self.optimal_duration = None
        self.constraints_active = None
        self.optimization_time = None
    
    def mean_variance_optimization(self, expected_returns: np.ndarray, 
                                covariance_matrix: np.ndarray,
                                current_weights: Optional[np.ndarray] = None,
                                constraints: Optional[Dict] = None) -> np.ndarray:
        """
        Perform mean-variance optimization.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            current_weights: Current portfolio weights (for transaction cost)
            constraints: Dictionary of constraints
            
        Returns:
            Array of optimal portfolio weights
        """
        logger.info("Performing mean-variance optimization")
        
        # Start timer
        start_time = time.time()
        
        # Number of assets
        n = len(expected_returns)
        
        # Define variables
        w = cp.Variable(n)
        
        # Define objective function
        ret = w @ expected_returns
        risk = cp.quad_form(w, covariance_matrix)
        
        # Add transaction costs if current weights are provided
        if current_weights is not None:
            transaction_cost_term = self.transaction_cost * cp.sum(cp.abs(w - current_weights))
            objective = ret - self.risk_aversion * risk - transaction_cost_term
        else:
            objective = ret - self.risk_aversion * risk
        
        # Define constraints
        constraint_list = [cp.sum(w) == 1]  # Fully invested
        
        # Handle additional constraints
        constraints_active = {'sum_to_one': True}
        
        if constraints:
            # Long-only constraint
            if constraints.get('long_only', False):
                constraint_list.append(w >= 0)
                constraints_active['long_only'] = True
            
            # Handle maximum weight constraint
            if 'max_weight' in constraints:
                max_weight = constraints['max_weight']
                constraint_list.append(w <= max_weight)
                constraints_active['max_weight'] = max_weight
            
            # Handle sector constraints
            if 'sector_constraints' in constraints and 'sector_mapping' in constraints:
                sector_constraints = constraints['sector_constraints']
                sector_mapping = constraints['sector_mapping']
                
                for sector, (min_weight, max_weight) in sector_constraints.items():
                    sector_indices = [i for i, s in enumerate(sector_mapping) if s == sector]
                    sector_weight = cp.sum(w[sector_indices])
                    
                    if min_weight is not None:
                        constraint_list.append(sector_weight >= min_weight)
                    
                    if max_weight is not None:
                        constraint_list.append(sector_weight <= max_weight)
                
                constraints_active['sector_constraints'] = True
            
            # Handle duration constraints
            if 'durations' in constraints and ('min_duration' in constraints or 'max_duration' in constraints):
                durations = constraints['durations']
                
                portfolio_duration = w @ durations
                
                if 'min_duration' in constraints:
                    min_duration = constraints['min_duration']
                    constraint_list.append(portfolio_duration >= min_duration)
                    constraints_active['min_duration'] = min_duration
                
                if 'max_duration' in constraints:
                    max_duration = constraints['max_duration']
                    constraint_list.append(portfolio_duration <= max_duration)
                    constraints_active['max_duration'] = max_duration
            
            # Handle rating constraints
            if 'rating_constraints' in constraints and 'rating_mapping' in constraints:
                rating_constraints = constraints['rating_constraints']
                rating_mapping = constraints['rating_mapping']
                
                for rating, (min_weight, max_weight) in rating_constraints.items():
                    rating_indices = [i for i, r in enumerate(rating_mapping) if r == rating]
                    rating_weight = cp.sum(w[rating_indices])
                    
                    if min_weight is not None:
                        constraint_list.append(rating_weight >= min_weight)
                    
                    if max_weight is not None:
                        constraint_list.append(rating_weight <= max_weight)
                
                constraints_active['rating_constraints'] = True
            
            # Handle tracking error constraint
            if 'benchmark_weights' in constraints and 'max_tracking_error' in constraints:
                benchmark_weights = constraints['benchmark_weights']
                max_tracking_error = constraints['max_tracking_error']
                
                tracking_error = cp.norm(w - benchmark_weights, 2)
                constraint_list.append(tracking_error <= max_tracking_error)
                
                constraints_active['max_tracking_error'] = max_tracking_error
            
            # Handle turnover constraint
            if 'max_turnover' in constraints and current_weights is not None:
                max_turnover = constraints['max_turnover']
                turnover = cp.sum(cp.abs(w - current_weights))
                constraint_list.append(turnover <= max_turnover)
                
                constraints_active['max_turnover'] = max_turnover
        
        # Define and solve the problem
        prob = cp.Problem(cp.Maximize(objective), constraint_list)
        
        try:
            prob.solve()
            
            if prob.status == 'optimal':
                logger.info(f"Optimization succeeded with optimal value: {prob.value:.6f}")
                
                # Store results
                self.optimal_weights = w.value
                self.expected_return = self.optimal_weights @ expected_returns
                self.expected_risk = np.sqrt(self.optimal_weights @ covariance_matrix @ self.optimal_weights)
                
                # Calculate portfolio duration if available
                if 'durations' in constraints:
                    self.optimal_duration = self.optimal_weights @ constraints['durations']
                else:
                    self.optimal_duration = None
                
                self.constraints_active = constraints_active
                self.optimization_time = time.time() - start_time
                
                return self.optimal_weights
            else:
                logger.error(f"Optimization failed with status: {prob.status}")
                return None
        except Exception as e:
            logger.error(f"Error during optimization: {e}")
            return None
    
    def risk_budgeting_optimization(self, covariance_matrix: np.ndarray, 
                                risk_budgets: np.ndarray,
                                current_weights: Optional[np.ndarray] = None,
                                constraints: Optional[Dict] = None) -> np.ndarray:
        """
        Perform risk budgeting optimization.
        
        Args:
            covariance_matrix: Covariance matrix of returns
            risk_budgets: Target risk contribution for each asset (proportional)
            current_weights: Current portfolio weights (for transaction cost)
            constraints: Dictionary of constraints
            
        Returns:
            Array of optimal portfolio weights
        """
        logger.info("Performing risk budgeting optimization")
        
        # Start timer
        start_time = time.time()
        
        # Number of assets
        n = len(risk_budgets)
        
        # Normalize risk budgets to sum to 1
        risk_budgets = risk_budgets / np.sum(risk_budgets)
        
        # Define objective function for risk budgeting
        def objective(w):
            # Normalize weights to sum to 1
            w = w / np.sum(w)
            
            # Calculate portfolio risk
            portfolio_risk = np.sqrt(w @ covariance_matrix @ w)
            
            # Calculate marginal contribution to risk (MCR)
            mcr = (covariance_matrix @ w) / portfolio_risk
            
            # Calculate risk contribution of each asset
            risk_contribution = w * mcr
            
            # Calculate deviation from target risk budgets
            deviation = risk_contribution - risk_budgets * portfolio_risk
            
            # Return sum of squared deviations
            return np.sum(deviation ** 2)
        
        # Initial guess (equal weights)
        initial_guess = np.ones(n) / n
        
        # Constraints for optimize
        constraints_list = []
        
        # Weights sum to 1
        constraints_list.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        
        # Handle additional constraints
        if constraints and constraints.get('long_only', False):
            bounds = [(0, None) for _ in range(n)]
        else:
            bounds = [(None, None) for _ in range(n)]
        
        # Minimize the objective
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )
        
        if result.success:
            # Normalize weights to sum to 1
            optimal_weights = result.x / np.sum(result.x)
            
            logger.info(f"Risk budgeting optimization succeeded with objective: {result.fun:.6f}")
            
            # Store results
            self.optimal_weights = optimal_weights
            self.expected_risk = np.sqrt(self.optimal_weights @ covariance_matrix @ self.optimal_weights)
            
            # Calculate portfolio duration if available
            if constraints and 'durations' in constraints:
                self.optimal_duration = self.optimal_weights @ constraints['durations']
            else:
                self.optimal_duration = None
            
            self.optimization_time = time.time() - start_time
            
            return self.optimal_weights
        else:
            logger.error(f"Risk budgeting optimization failed: {result.message}")
            return None
    
    def duration_matching_optimization(self, expected_returns: np.ndarray, 
                                     covariance_matrix: np.ndarray,
                                     durations: np.ndarray,
                                     target_duration: float,
                                     current_weights: Optional[np.ndarray] = None,
                                     constraints: Optional[Dict] = None) -> np.ndarray:
        """
        Perform duration-matching optimization.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            durations: Modified duration of each asset
            target_duration: Target portfolio duration
            current_weights: Current portfolio weights (for transaction cost)
            constraints: Dictionary of constraints
            
        Returns:
            Array of optimal portfolio weights
        """
        logger.info(f"Performing duration-matching optimization with target duration {target_duration}")
        
        # Start timer
        start_time = time.time()
        
        # Number of assets
        n = len(expected_returns)
        
        # Define variables
        w = cp.Variable(n)
        
        # Define objective function
        ret = w @ expected_returns
        risk = cp.quad_form(w, covariance_matrix)
        
        # Add transaction costs if current weights are provided
        if current_weights is not None:
            transaction_cost_term = self.transaction_cost * cp.sum(cp.abs(w - current_weights))
            objective = ret - self.risk_aversion * risk - transaction_cost_term
        else:
            objective = ret - self.risk_aversion * risk
        
        # Define constraints
        constraint_list = [
            cp.sum(w) == 1,  # Fully invested
            w @ durations == target_duration  # Duration matching
        ]
        
        # Handle additional constraints
        constraints_active = {'sum_to_one': True, 'target_duration': target_duration}
        
        if constraints:
            # Long-only constraint
            if constraints.get('long_only', False):
                constraint_list.append(w >= 0)
                constraints_active['long_only'] = True
            
            # Handle maximum weight constraint
            if 'max_weight' in constraints:
                max_weight = constraints['max_weight']
                constraint_list.append(w <= max_weight)
                constraints_active['max_weight'] = max_weight
            
            # Handle sector constraints
            if 'sector_constraints' in constraints and 'sector_mapping' in constraints:
                sector_constraints = constraints['sector_constraints']
                sector_mapping = constraints['sector_mapping']
                
                for sector, (min_weight, max_weight) in sector_constraints.items():
                    sector_indices = [i for i, s in enumerate(sector_mapping) if s == sector]
                    sector_weight = cp.sum(w[sector_indices])
                    
                    if min_weight is not None:
                        constraint_list.append(sector_weight >= min_weight)
                    
                    if max_weight is not None:
                        constraint_list.append(sector_weight <= max_weight)
                
                constraints_active['sector_constraints'] = True
            
            # Handle convexity constraints
            if 'convexities' in constraints and ('min_convexity' in constraints or 'max_convexity' in constraints):
                convexities = constraints['convexities']
                
                portfolio_convexity = w @ convexities
                
                if 'min_convexity' in constraints:
                    min_convexity = constraints['min_convexity']
                    constraint_list.append(portfolio_convexity >= min_convexity)
                    constraints_active['min_convexity'] = min_convexity
                
                if 'max_convexity' in constraints:
                    max_convexity = constraints['max_convexity']
                    constraint_list.append(portfolio_convexity <= max_convexity)
                    constraints_active['max_convexity'] = max_convexity
            
            # Handle rating constraints
            if 'rating_constraints' in constraints and 'rating_mapping' in constraints:
                rating_constraints = constraints['rating_constraints']
                rating_mapping = constraints['rating_mapping']
                
                for rating, (min_weight, max_weight) in rating_constraints.items():
                    rating_indices = [i for i, r in enumerate(rating_mapping) if r == rating]
                    rating_weight = cp.sum(w[rating_indices])
                    
                    if min_weight is not None:
                        constraint_list.append(rating_weight >= min_weight)
                    
                    if max_weight is not None:
                        constraint_list.append(rating_weight <= max_weight)
                
                constraints_active['rating_constraints'] = True
        
        # Define and solve the problem
        prob = cp.Problem(cp.Maximize(objective), constraint_list)
        
        try:
            prob.solve()
            
            if prob.status == 'optimal':
                logger.info(f"Duration-matching optimization succeeded with optimal value: {prob.value:.6f}")
                
                # Store results
                self.optimal_weights = w.value
                self.expected_return = self.optimal_weights @ expected_returns
                self.expected_risk = np.sqrt(self.optimal_weights @ covariance_matrix @ self.optimal_weights)
                self.optimal_duration = self.optimal_weights @ durations
                
                self.constraints_active = constraints_active
                self.optimization_time = time.time() - start_time
                
                return self.optimal_weights
            else:
                logger.error(f"Duration-matching optimization failed with status: {prob.status}")
                return None
        except Exception as e:
            logger.error(f"Error during duration-matching optimization: {e}")
            return None
    
    def maximum_sharpe_ratio(self, expected_returns: np.ndarray, 
                           covariance_matrix: np.ndarray,
                           risk_free_rate: float = 0.0,
                           current_weights: Optional[np.ndarray] = None,
                           constraints: Optional[Dict] = None) -> np.ndarray:
        """
        Find the portfolio with the maximum Sharpe ratio.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            risk_free_rate: Risk-free rate
            current_weights: Current portfolio weights (for transaction cost)
            constraints: Dictionary of constraints
            
        Returns:
            Array of optimal portfolio weights
        """
        logger.info(f"Finding maximum Sharpe ratio portfolio with risk-free rate {risk_free_rate}")
        
        # Start timer
        start_time = time.time()
        
        # Number of assets
        n = len(expected_returns)
        
        # Define variables
        w = cp.Variable(n)
        
        # Define auxiliary variable for the denominator (standard deviation)
        risk = cp.sqrt(cp.quad_form(w, covariance_matrix))
        
        # Define objective function
        excess_return = w @ expected_returns - risk_free_rate
        
        # Define constraints
        constraint_list = [cp.sum(w) == 1]  # Fully invested
        
        # Handle additional constraints
        constraints_active = {'sum_to_one': True}
        
        if constraints:
            # Long-only constraint
            if constraints.get('long_only', False):
                constraint_list.append(w >= 0)
                constraints_active['long_only'] = True
            
            # Handle maximum weight constraint
            if 'max_weight' in constraints:
                max_weight = constraints['max_weight']
                constraint_list.append(w <= max_weight)
                constraints_active['max_weight'] = max_weight
            
            # Handle sector constraints
            if 'sector_constraints' in constraints and 'sector_mapping' in constraints:
                sector_constraints = constraints['sector_constraints']
                sector_mapping = constraints['sector_mapping']
                
                for sector, (min_weight, max_weight) in sector_constraints.items():
                    sector_indices = [i for i, s in enumerate(sector_mapping) if s == sector]
                    sector_weight = cp.sum(w[sector_indices])
                    
                    if min_weight is not None:
                        constraint_list.append(sector_weight >= min_weight)
                    
                    if max_weight is not None:
                        constraint_list.append(sector_weight <= max_weight)
                
                constraints_active['sector_constraints'] = True
            
            # Handle duration constraints
            if 'durations' in constraints and ('min_duration' in constraints or 'max_duration' in constraints):
                durations = constraints['durations']
                
                portfolio_duration = w @ durations
                
                if 'min_duration' in constraints:
                    min_duration = constraints['min_duration']
                    constraint_list.append(portfolio_duration >= min_duration)
                    constraints_active['min_duration'] = min_duration
                
                if 'max_duration' in constraints:
                    max_duration = constraints['max_duration']
                    constraint_list.append(portfolio_duration <= max_duration)
                    constraints_active['max_duration'] = max_duration
        
        # Define and solve the problem
        # We maximize excess_return / risk, but this is non-convex
        # Instead, we maximize excess_return subject to risk <= 1
        constraint_list.append(risk <= 1)
        prob = cp.Problem(cp.Maximize(excess_return), constraint_list)
        
        try:
            prob.solve()
            
            if prob.status == 'optimal':
                # Scale the weights to get actual solution
                scale = 1.0 / risk.value
                optimal_weights = w.value * scale
                
                logger.info(f"Maximum Sharpe ratio optimization succeeded with Sharpe ratio: {prob.value:.6f}")
                
                # Store results
                self.optimal_weights = optimal_weights
                self.expected_return = self.optimal_weights @ expected_returns
                self.expected_risk = np.sqrt(self.optimal_weights @ covariance_matrix @ self.optimal_weights)
                
                # Calculate portfolio duration if available
                if constraints and 'durations' in constraints:
                    self.optimal_duration = self.optimal_weights @ constraints['durations']
                else:
                    self.optimal_duration = None
                
                self.constraints_active = constraints_active
                self.optimization_time = time.time() - start_time
                
                return self.optimal_weights
            else:
                logger.error(f"Maximum Sharpe ratio optimization failed with status: {prob.status}")
                return None
        except Exception as e:
            logger.error(f"Error during maximum Sharpe ratio optimization: {e}")
            return None
    
    def minimum_tracking_error(self, covariance_matrix: np.ndarray,
                             benchmark_weights: np.ndarray,
                             current_weights: Optional[np.ndarray] = None,
                             expected_returns: Optional[np.ndarray] = None,
                             alpha: float = 0.0,
                             constraints: Optional[Dict] = None) -> np.ndarray:
        """
        Find the portfolio with minimum tracking error to a benchmark.
        
        Args:
            covariance_matrix: Covariance matrix of returns
            benchmark_weights: Benchmark portfolio weights
            current_weights: Current portfolio weights (for transaction cost)
            expected_returns: Expected returns for each asset (optional)
            alpha: Weight on expected return in the objective (0 = pure tracking error)
            constraints: Dictionary of constraints
            
        Returns:
            Array of optimal portfolio weights
        """
        logger.info("Finding portfolio with minimum tracking error")
        
        # Start timer
        start_time = time.time()
        
        # Number of assets
        n = len(benchmark_weights)
        
        # Define variables
        w = cp.Variable(n)
        
        # Define objective function
        tracking_error = cp.quad_form(w - benchmark_weights, covariance_matrix)
        
        if expected_returns is not None and alpha > 0:
            # Include expected return component
            ret = w @ expected_returns
            objective = -alpha * ret + tracking_error
        else:
            objective = tracking_error
        
        # Add transaction costs if current weights are provided
        if current_weights is not None:
            transaction_cost_term = self.transaction_cost * cp.sum(cp.abs(w - current_weights))
            objective += transaction_cost_term
        
        # Define constraints
        constraint_list = [cp.sum(w) == 1]  # Fully invested
        
        # Handle additional constraints
        constraints_active = {'sum_to_one': True}
        
        if constraints:
            # Long-only constraint
            if constraints.get('long_only', False):
                constraint_list.append(w >= 0)
                constraints_active['long_only'] = True
            
            # Handle maximum weight constraint
            if 'max_weight' in constraints:
                max_weight = constraints['max_weight']
                constraint_list.append(w <= max_weight)
                constraints_active['max_weight'] = max_weight
            
            # Handle sector constraints
            if 'sector_constraints' in constraints and 'sector_mapping' in constraints:
                sector_constraints = constraints['sector_constraints']
                sector_mapping = constraints['sector_mapping']
                
                for sector, (min_weight, max_weight) in sector_constraints.items():
                    sector_indices = [i for i, s in enumerate(sector_mapping) if s == sector]
                    sector_weight = cp.sum(w[sector_indices])
                    
                    if min_weight is not None:
                        constraint_list.append(sector_weight >= min_weight)
                    
                    if max_weight is not None:
                        constraint_list.append(sector_weight <= max_weight)
                
                constraints_active['sector_constraints'] = True
            
            # Handle duration constraints
            if 'durations' in constraints and ('min_duration' in constraints or 'max_duration' in constraints):
                durations = constraints['durations']
                
                portfolio_duration = w @ durations
                
                if 'min_duration' in constraints:
                    min_duration = constraints['min_duration']
                    constraint_list.append(portfolio_duration >= min_duration)
                    constraints_active['min_duration'] = min_duration
                
                if 'max_duration' in constraints:
                    max_duration = constraints['max_duration']
                    constraint_list.append(portfolio_duration <= max_duration)
                    constraints_active['max_duration'] = max_duration
        
        # Define and solve the problem
        prob = cp.Problem(cp.Minimize(objective), constraint_list)
        
        try:
            prob.solve()
            
            if prob.status == 'optimal':
                logger.info(f"Minimum tracking error optimization succeeded with optimal value: {prob.value:.6f}")
                
                # Store results
                self.optimal_weights = w.value
                
                if expected_returns is not None:
                    self.expected_return = self.optimal_weights @ expected_returns
                else:
                    self.expected_return = None
                
                self.expected_risk = np.sqrt(self.optimal_weights @ covariance_matrix @ self.optimal_weights)
                
                # Calculate portfolio duration if available
                if constraints and 'durations' in constraints:
                    self.optimal_duration = self.optimal_weights @ constraints['durations']
                else:
                    self.optimal_duration = None
                
                self.constraints_active = constraints_active
                self.optimization_time = time.time() - start_time
                
                return self.optimal_weights
            else:
                logger.error(f"Minimum tracking error optimization failed with status: {prob.status}")
                return None
        except Exception as e:
            logger.error(f"Error during minimum tracking error optimization: {e}")
            return None
    
    def transaction_cost_optimization(self, expected_returns: np.ndarray, 
                                    covariance_matrix: np.ndarray,
                                    current_weights: np.ndarray,
                                    transaction_costs: np.ndarray,
                                    target_risk: Optional[float] = None,
                                    constraints: Optional[Dict] = None) -> np.ndarray:
        """
        Optimize portfolio considering transaction costs.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            current_weights: Current portfolio weights
            transaction_costs: Transaction cost for each asset (as a fraction)
            target_risk: Target portfolio risk (if None, use mean-variance with risk aversion)
            constraints: Dictionary of constraints
            
        Returns:
            Array of optimal portfolio weights
        """
        logger.info("Performing transaction cost optimization")
        
        # Start timer
        start_time = time.time()
        
        # Number of assets
        n = len(expected_returns)
        
        # Define variables
        w = cp.Variable(n)
        
        # Define objective function components
        ret = w @ expected_returns
        risk = cp.quad_form(w, covariance_matrix)
        
        # Transaction cost term (asset-specific costs)
        trade_sizes = cp.abs(w - current_weights)
        transaction_cost_term = trade_sizes @ transaction_costs
        
        # Combine objective
        if target_risk is None:
            # Mean-variance with risk aversion
            objective = ret - self.risk_aversion * risk - transaction_cost_term
            problem_type = 'maximize'
        else:
            # Maximize return for target risk
            objective = ret - transaction_cost_term
            problem_type = 'maximize'
        
        # Define constraints
        constraint_list = [cp.sum(w) == 1]  # Fully invested
        
        # Add target risk constraint if specified
        if target_risk is not None:
            constraint_list.append(cp.sqrt(risk) <= target_risk)
        
        # Handle additional constraints
        constraints_active = {'sum_to_one': True}
        
        if constraints:
            # Long-only constraint
            if constraints.get('long_only', False):
                constraint_list.append(w >= 0)
                constraints_active['long_only'] = True
            
            # Handle maximum weight constraint
            if 'max_weight' in constraints:
                max_weight = constraints['max_weight']
                constraint_list.append(w <= max_weight)
                constraints_active['max_weight'] = max_weight
            
            # Handle sector constraints
            if 'sector_constraints' in constraints and 'sector_mapping' in constraints:
                sector_constraints = constraints['sector_constraints']
                sector_mapping = constraints['sector_mapping']
                
                for sector, (min_weight, max_weight) in sector_constraints.items():
                    sector_indices = [i for i, s in enumerate(sector_mapping) if s == sector]
                    sector_weight = cp.sum(w[sector_indices])
                    
                    if min_weight is not None:
                        constraint_list.append(sector_weight >= min_weight)
                    
                    if max_weight is not None:
                        constraint_list.append(sector_weight <= max_weight)
                
                constraints_active['sector_constraints'] = True
            
            # Handle duration constraints
            if 'durations' in constraints and ('min_duration' in constraints or 'max_duration' in constraints):
                durations = constraints['durations']
                
                portfolio_duration = w @ durations
                
                if 'min_duration' in constraints:
                    min_duration = constraints['min_duration']
                    constraint_list.append(portfolio_duration >= min_duration)
                    constraints_active['min_duration'] = min_duration
                
                if 'max_duration' in constraints:
                    max_duration = constraints['max_duration']
                    constraint_list.append(portfolio_duration <= max_duration)
                    constraints_active['max_duration'] = max_duration
            
            # Handle maximum turnover constraint
            if 'max_turnover' in constraints:
                max_turnover = constraints['max_turnover']
                turnover = cp.sum(trade_sizes)
                constraint_list.append(turnover <= max_turnover)
                constraints_active['max_turnover'] = max_turnover
        
        # Define and solve the problem
        if problem_type == 'maximize':
            prob = cp.Problem(cp.Maximize(objective), constraint_list)
        else:
            prob = cp.Problem(cp.Minimize(objective), constraint_list)
        
        try:
            prob.solve()
            
            if prob.status == 'optimal':
                logger.info(f"Transaction cost optimization succeeded with optimal value: {prob.value:.6f}")
                
                # Store results
                self.optimal_weights = w.value
                self.expected_return = self.optimal_weights @ expected_returns
                self.expected_risk = np.sqrt(self.optimal_weights @ covariance_matrix @ self.optimal_weights)
                
                # Calculate portfolio duration if available
                if constraints and 'durations' in constraints:
                    self.optimal_duration = self.optimal_weights @ constraints['durations']
                else:
                    self.optimal_duration = None
                
                # Calculate transaction costs
                self.transaction_cost_value = np.sum(np.abs(self.optimal_weights - current_weights) * transaction_costs)
                
                self.constraints_active = constraints_active
                self.optimization_time = time.time() - start_time
                
                return self.optimal_weights
            else:
                logger.error(f"Transaction cost optimization failed with status: {prob.status}")
                return None
        except Exception as e:
            logger.error(f"Error during transaction cost optimization: {e}")
            return None
    
    def compute_portfolio_metrics(self, weights: np.ndarray, 
                               expected_returns: np.ndarray, 
                               covariance_matrix: np.ndarray,
                               risk_free_rate: float = 0.0) -> Dict[str, float]:
        """
        Compute various portfolio metrics.
        
        Args:
            weights: Portfolio weights
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            risk_free_rate: Risk-free rate
            
        Returns:
            Dictionary with portfolio metrics
        """
        # Portfolio expected return
        portfolio_return = weights @ expected_returns
        
        # Portfolio volatility
        portfolio_volatility = np.sqrt(weights @ covariance_matrix @ weights)
        
        # Sharpe ratio
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        
        # Asset contributions to risk
        marginal_contribution = covariance_matrix @ weights
        risk_contribution = weights * marginal_contribution / portfolio_volatility
        
        # Diversification ratio
        weighted_vol = weights @ np.sqrt(np.diag(covariance_matrix))
        diversification_ratio = weighted_vol / portfolio_volatility
        
        # Return metrics
        metrics = {
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'diversification_ratio': diversification_ratio
        }
        
        return metrics
    
    def plot_efficient_frontier(self, expected_returns: np.ndarray, 
                              covariance_matrix: np.ndarray,
                              constraints: Optional[Dict] = None,
                              risk_free_rate: float = 0.0,
                              n_points: int = 50) -> plt.Figure:
        """
        Plot the efficient frontier.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            constraints: Dictionary of constraints
            risk_free_rate: Risk-free rate
            n_points: Number of points on the efficient frontier
            
        Returns:
            Matplotlib figure
        """
        logger.info("Plotting efficient frontier")
        
        # Store original risk aversion
        original_risk_aversion = self.risk_aversion
        
        # Generate range of risk aversion parameters
        risk_aversions = np.logspace(-2, 3, n_points)
        
        # Initialize arrays for results
        returns = np.zeros(n_points)
        volatilities = np.zeros(n_points)
        
        # Compute portfolios for different risk aversions
        for i, risk_aversion in enumerate(risk_aversions):
            self.risk_aversion = risk_aversion
            
            weights = self.mean_variance_optimization(
                expected_returns=expected_returns,
                covariance_matrix=covariance_matrix,
                constraints=constraints
            )
            
            if weights is not None:
                returns[i] = self.expected_return
                volatilities[i] = self.expected_risk
            else:
                returns[i] = np.nan
                volatilities[i] = np.nan
        
        # Reset risk aversion
        self.risk_aversion = original_risk_aversion
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot efficient frontier
        ax.plot(volatilities, returns, 'b-', linewidth=2, label='Efficient Frontier')
        
        # Compute and plot maximum Sharpe ratio portfolio
        max_sharpe_weights = self.maximum_sharpe_ratio(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            risk_free_rate=risk_free_rate,
            constraints=constraints
        )
        
        if max_sharpe_weights is not None:
            max_sharpe_return = self.expected_return
            max_sharpe_risk = self.expected_risk
            ax.scatter(max_sharpe_risk, max_sharpe_return, marker='*', color='r', s=100, label='Maximum Sharpe Ratio')
        
        # Reset optimization
        self.risk_aversion = original_risk_aversion
        self.mean_variance_optimization(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            constraints=constraints
        )
        
        if self.optimal_weights is not None:
            ax.scatter(self.expected_risk, self.expected_return, marker='o', color='g', s=100, label='Current Portfolio')
        
        # Add risk-free rate
        ax.axhline(risk_free_rate, color='k', linestyle='--', linewidth=1)
        
        # Add optimal Capital Allocation Line
        if max_sharpe_weights is not None:
            # Plot CAL
            x_range = np.linspace(0, max(volatilities) * 1.2, 100)
            sharpe = (max_sharpe_return - risk_free_rate) / max_sharpe_risk
            y_range = risk_free_rate + sharpe * x_range
            ax.plot(x_range, y_range, 'r--', linewidth=1, label='Capital Allocation Line')
        
        # Add labels and legend
        ax.set_xlabel('Portfolio Volatility')
        ax.set_ylabel('Portfolio Expected Return')
        ax.set_title('Efficient Frontier')
        ax.legend()
        ax.grid(True)
        
        return fig
