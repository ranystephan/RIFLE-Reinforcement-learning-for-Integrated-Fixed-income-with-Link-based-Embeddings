import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from datetime import datetime, timedelta
from scipy.optimize import minimize, fsolve
from scipy.stats import norm
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InterestRateModel:
    """
    Class for simulating interest rate processes and yield curve evolution.
    """
    
    def __init__(self, model_type: str = 'vasicek', random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        np.random.seed(random_state)
        self.params = {}           # Calibrated parameters
        self.historical_data = None
        self.regimes = None
        self.regime_params = {}
    
    def vasicek_process(self, r0: float, kappa: float, theta: float, sigma: float, 
                         T: float, dt: float) -> np.ndarray:
        steps = int(T / dt)
        rates = np.zeros(steps + 1)
        rates[0] = r0
        for t in range(1, steps + 1):
            dW = np.random.normal(0, np.sqrt(dt))
            dr = kappa * (theta - rates[t-1]) * dt + sigma * dW
            rates[t] = rates[t-1] + dr
        return rates
    
    def cir_process(self, r0: float, kappa: float, theta: float, sigma: float, 
                    T: float, dt: float) -> np.ndarray:
        steps = int(T / dt)
        rates = np.zeros(steps + 1)
        rates[0] = r0
        for t in range(1, steps + 1):
            dW = np.random.normal(0, np.sqrt(dt))
            # FIX: Ensure non-negative square root argument and final rates non-negative.
            dr = kappa * (theta - rates[t-1]) * dt + sigma * np.sqrt(max(rates[t-1], 0)) * dW
            rates[t] = max(rates[t-1] + dr, 0)
        return rates
    
    def hull_white_process(self, r0: float, kappa: float, sigma: float, 
                           theta_t: Callable[[float], float], T: float, dt: float) -> np.ndarray:
        steps = int(T/dt)
        rates = np.zeros(steps+1)
        rates[0] = r0
        # FIX: Using Euler discretization and allowing negative rates (HW rates can be negative)
        for t in range(1, steps+1):
            time = t*dt
            dW = np.random.normal(0, np.sqrt(dt))
            drift = kappa * (theta_t(time) - rates[t-1]) * dt
            diffusion = sigma * dW
            rates[t] = rates[t-1] + drift + diffusion
        return rates
        
    def calibrate(self, historical_rates: pd.DataFrame, tenor: str = '3M'):
        """
        Calibrate the interest rate model to historical data using robust methods
        with realistic constraints on parameters.
        
        Args:
            historical_rates: DataFrame with historical rates
            tenor: Tenor to use for calibration
        """
        logger.info(f"Calibrating {self.model_type} model to historical data")
        self.historical_data = historical_rates
        
        # Ensure we have valid data
        if tenor in historical_rates.columns:
            rates = historical_rates[tenor].values
        else:
            available_cols = [col for col in historical_rates.columns if isinstance(col, str) and any(t in col for t in ['Y', 'M'])]
            if available_cols:
                tenor = available_cols[0]
                rates = historical_rates[tenor].values
                logger.warning(f"Tenor {tenor} not found. Using {tenor} instead.")
            else:
                raise ValueError(f"No valid rate columns found in historical data")
        
        dt = 1/252  # daily time step
        rate_changes = np.diff(rates)
        
        if self.model_type == 'vasicek':
            # Improved calibration with realistic parameter constraints
            # Use robust method-of-moments estimation with bounds
            X = rates[:-1].reshape(-1, 1)
            y = rate_changes
            
            # Calculate statistics for method of moments
            mean_rate = np.mean(rates)
            std_rate = np.std(rates)
            auto_corr = np.corrcoef(rates[:-1], rates[1:])[0, 1]
            
            # Estimate kappa (mean reversion speed) - constrain to reasonable values
            kappa_est = -np.log(auto_corr) / dt
            kappa = min(max(kappa_est, 0.05), 5.0)  # Constrain between 0.05 and 5.0
            
            # Estimate theta (long-term mean) - constrain to reasonable values
            beta = np.cov(X.flatten(), y)[0, 1] / np.var(X.flatten())
            alpha = np.mean(y) - beta * np.mean(X)
            theta_est = -alpha / beta if beta != 0 else mean_rate
            theta = min(max(theta_est, 0.005), 0.08)  # Constrain between 0.5% and 8%
            
            # Estimate sigma (volatility) - constrain to reasonable values
            sigma_est = np.sqrt(np.var(rate_changes) / dt)
            sigma = min(max(sigma_est, 0.001), 0.05)  # Constrain between 0.1% and 5%
            
            self.params = {
                'kappa': kappa,
                'theta': theta,
                'sigma': sigma,
                'r0': min(max(rates[-1], 0.001), 0.1)  # Constrain r0 to reasonable values
            }
            
        elif self.model_type == 'cir':
            # Improved CIR calibration with parameter constraints
            def objective(params):
                kappa, theta, sigma = params
                if kappa <= 0 or theta <= 0 or sigma <= 0:
                    return 1e10
                
                # Penalize unrealistic parameter values
                penalty = 0
                if kappa < 0.05 or kappa > 5.0:
                    penalty += 1e5
                if theta < 0.005 or theta > 0.08:
                    penalty += 1e5
                if sigma < 0.001 or sigma > 0.05:
                    penalty += 1e5
                
                try:
                    simulated = self.cir_process(
                        r0=rates[0],
                        kappa=kappa,
                        theta=theta,
                        sigma=sigma,
                        T=len(rates)*dt,
                        dt=dt
                    )
                    mse = np.mean((simulated[1:] - rates[1:])**2)
                    return mse + penalty
                except:
                    return 1e10
            
            # Use realistic initial guesses
            initial_guess = [0.5, np.mean(rates), 0.01]
            bounds = [(0.05, 5.0), (0.005, 0.08), (0.001, 0.05)]
            
            result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
            kappa, theta, sigma = result.x
            
            self.params = {
                'kappa': kappa,
                'theta': theta,
                'sigma': sigma,
                'r0': min(max(rates[-1], 0.001), 0.1)  # Constrain r0 to reasonable values
            }
            
        elif self.model_type == 'hw':
            # FIX: Use a cubic spline discount curve and calibrate a Vasicek-like forward curve.
            from scipy.interpolate import CubicSpline
            from scipy.optimize import curve_fit
            market_tenors = np.array([0.25, 0.5, 1, 2, 5, 10, 30])
            market_yields = np.array([0.01, 0.015, 0.018, 0.02, 0.025, 0.03, 0.035])
            discount_factors = np.exp(-market_yields * market_tenors)
            discount_curve = CubicSpline(market_tenors, discount_factors)
            t_points = np.linspace(0.1, 10, 100)  # avoid t=0 for stability
            # Compute instantaneous forward rates via numerical differentiation
            fwd_rates = -np.gradient(np.log(discount_curve(t_points)), t_points)
            def vasicek_curve(t, kappa, theta):
                return theta + (fwd_rates[0] - theta) * np.exp(-kappa*t)
            params_fit, _ = curve_fit(vasicek_curve, t_points, fwd_rates, p0=[0.1, 0.03], bounds=(0, [1, 0.1]))
            kappa, theta = params_fit
            # FIX: Estimate sigma from historical rate changes.
            sigma = np.std(rate_changes) / np.sqrt(dt) if len(rate_changes) > 0 else 0.01
            def theta_func(t):
                df = discount_curve(t)
                d2f = discount_curve(t, 2)
                return (d2f/df) + kappa*(fwd_rates[0] - theta)*np.exp(-kappa*t) + 0.5*sigma**2
            self.params = {
                'kappa': kappa,
                'theta_t': theta_func,
                'sigma': sigma,
                'r0': market_yields[0]
            }
        logger.info(f"Calibrated parameters: {self.params}")
    
    def calibrate_regimes(self, historical_rates: pd.DataFrame, regimes: pd.DataFrame, tenor: str = '3M'):
        logger.info(f"Calibrating {self.model_type} model for different regimes")
        self.regimes = regimes
        self.regime_params = {}
        unique_regimes = regimes['regime'].unique()
        for regime in unique_regimes:
            regime_dates = regimes[regimes['regime'] == regime].index
            regime_rates = historical_rates.loc[historical_rates.index.isin(regime_dates)]
            try:
                self.calibrate(regime_rates, tenor)
                self.regime_params[regime] = self.params.copy()
                logger.info(f"Calibrated parameters for regime {regime}: {self.params}")
            except Exception as e:
                logger.error(f"Error calibrating for regime {regime}: {e}")
    
    def simulate(self, T: float, dt: float, regime: Optional[int] = None) -> np.ndarray:
        if regime is not None and regime in self.regime_params:
            params = self.regime_params[regime]
        else:
            params = self.params
        if not params:
            raise ValueError("Model not calibrated. Call calibrate() first.")
        if self.model_type == 'vasicek':
            return self.vasicek_process(r0=params['r0'], kappa=params['kappa'], theta=params['theta'], sigma=params['sigma'], T=T, dt=dt)
        elif self.model_type == 'cir':
            return self.cir_process(r0=params['r0'], kappa=params['kappa'], theta=params['theta'], sigma=params['sigma'], T=T, dt=dt)
        elif self.model_type == 'hw':
            return self.hull_white_process(r0=params['r0'], kappa=params['kappa'], sigma=params['sigma'], theta_t=params['theta_t'], T=T, dt=dt)
    
    def simulate_yield_curve(self, short_rates: np.ndarray, tenors: List[float], model_type: str = 'nss') -> np.ndarray:
        from scipy.optimize import curve_fit
        def ns_model(t, beta0, beta1, beta2, tau):
            return beta0 + beta1*(1-np.exp(-t/tau))/(t/tau) + beta2*((1-np.exp(-t/tau))/(t/tau)-np.exp(-t/tau))
        market_tenors = np.array([0.25, 0.5, 1, 2, 5, 10, 30])
        market_yields = short_rates[-1] + 0.02*(1 - np.exp(-market_tenors/10))
        params, _ = curve_fit(ns_model, market_tenors, market_yields, p0=[0.03, -0.02, 0.01, 2.0], bounds=([0.01,-0.1,-0.1,0.1], [0.1,0.1,0.1,10]))
        beta0, beta1, beta2, tau = params
        yield_curves = np.zeros((len(short_rates), len(tenors)))
        for t_idx, r_t in enumerate(short_rates):
            adj_beta0 = r_t + (beta0 - np.mean(short_rates))
            adj_beta1 = beta1 * (1 + 0.1*np.random.randn())
            adj_tau = tau * (1 + 0.05*np.random.randn())
            for tenor_idx, tenor in enumerate(tenors):
                if tenor == 0:
                    yield_curves[t_idx, tenor_idx] = r_t
                else:
                    term = (1 - np.exp(-tenor/adj_tau))/(tenor/adj_tau)
                    yield_curves[t_idx, tenor_idx] = adj_beta0 + adj_beta1*term + beta2*(term - np.exp(-tenor/adj_tau))
        return yield_curves


class CreditSpreadModel:
    """
    Class for simulating credit spread dynamics.
    """
    
    def __init__(self, model_type: str = 'merton', random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        np.random.seed(random_state)
        self.params = {}
        self.regimes = None
        self.regime_params = {}
    
    def merton_spread(self, asset_value: float, risk_free_rate: float, T: float) -> float:
        # FIX: Updated signature to include risk_free_rate and T.
        d1 = (np.log(asset_value / self.params['debt_value']) + 
              (risk_free_rate + 0.5 * self.params['asset_vol']**2) * T) / (self.params['asset_vol'] * np.sqrt(T))
        pd = norm.cdf(-d1)  # probability of default
        lgd = 1 - self.params['recovery_rate']
        # Credit spread computed from default probability and loss given default
        return -np.log(1 - pd * lgd) / T if T > 0 else 0
    
    def simulate_merton(self, asset_value: float, debt_value: float, 
                        asset_vol: float, T: float, dt: float, 
                        r_process: np.ndarray) -> np.ndarray:
        steps = int(T / dt)
        spreads = np.zeros(steps + 1)
        asset_process = np.zeros(steps + 1)
        asset_process[0] = asset_value
        for t in range(1, steps + 1):
            dW = np.random.normal(0, np.sqrt(dt))
            asset_process[t] = asset_process[t-1] * np.exp((r_process[t-1] - 0.5 * asset_vol**2) * dt + asset_vol * dW)
        for t in range(steps + 1):
            time_to_maturity = T - t * dt
            if time_to_maturity <= 0:
                spreads[t] = 0
            else:
                spreads[t] = self.merton_spread(asset_value=asset_process[t], risk_free_rate=r_process[t], T=time_to_maturity)
        return spreads
    
    def calibrate(self, corporate_df, yields_df=None, tenor='10Y', rating='BAA'):
        if self.model_type == 'merton':
            equity_value = corporate_df['market_cap'].mean()
            equity_vol = corporate_df['equity_vol'].mean()
            debt_value = corporate_df['total_debt'].mean()
            risk_free_rate = yields_df[tenor].mean() if yields_df is not None else 0.03
            T = 1.0  # horizon of 1 year
            def equations(params):
                A, sigma_A = params
                d1 = (np.log(A/debt_value) + (risk_free_rate + 0.5 * sigma_A**2) * T) / (sigma_A * np.sqrt(T))
                d2 = d1 - sigma_A * np.sqrt(T)
                eq1 = A * norm.cdf(d1) - debt_value * np.exp(-risk_free_rate * T) * norm.cdf(d2) - equity_value
                eq2 = equity_vol * equity_value - A * sigma_A * norm.cdf(d1)
                return [eq1, eq2]
            A0 = equity_value + debt_value
            sigma_A0 = 0.2
            solution = fsolve(equations, [A0, sigma_A0])
            self.params = {
                'asset_value': solution[0],
                'asset_vol': solution[1],
                'debt_value': debt_value,
                'T': T,
                'recovery_rate': 0.4  # historical estimate
            }
            logger.info(f"Calibrated Merton model parameters: {self.params}")
        elif self.model_type == 'simple':
            if f"{rating}_spread" in corporate_df.columns:
                mean_spread = corporate_df[f"{rating}_spread"].mean() / 100
                vol_spread = corporate_df[f"{rating}_spread"].std() / 100
            elif rating in corporate_df.columns:
                mean_spread = corporate_df[rating].mean() / 100
                vol_spread = corporate_df[rating].std() / 100
            else:
                rating_cols = [col for col in corporate_df.columns if rating in col]
                if rating_cols:
                    mean_spread = corporate_df[rating_cols[0]].mean() / 100
                    vol_spread = corporate_df[rating_cols[0]].std() / 100
                else:
                    numeric_cols = corporate_df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        mean_spread = corporate_df[numeric_cols[0]].mean() / 100
                        vol_spread = corporate_df[numeric_cols[0]].std() / 100
                    else:
                        raise ValueError(f"No suitable data found for {rating} calibration")
            self.params = {
                'mean_spread': mean_spread,
                'vol_spread': vol_spread,
                'rating': rating,
                'tenor': tenor
            }
            logger.info(f"Calibrated simple model parameters: {self.params}")
    
    def calibrate_regimes(self, corporate_df, yields_df, regime_df, tenor='10Y', rating='BAA'):
        logger.info(f"Calibrating {self.model_type} model for different regimes")
        logger.info(f"Corporate DataFrame columns: {corporate_df.columns.tolist()}")
        spread_col = f"{rating}_{tenor}_spread"
        if spread_col not in corporate_df.columns:
            if rating in corporate_df.columns and tenor in yields_df.columns:
                corporate_df[spread_col] = corporate_df[rating] - yields_df[tenor]
                logger.info(f"Created {spread_col} from {rating} and {tenor} columns")
            elif 'BBB' in corporate_df.columns and tenor in yields_df.columns:
                corporate_df[spread_col] = corporate_df['BBB'] - yields_df[tenor]
                logger.info(f"Created {spread_col} using BBB as substitute")
            elif 'BAA' in corporate_df.columns and '10Y' in yields_df.columns:
                corporate_df['BAA_10Y_spread'] = corporate_df['BAA'] - yields_df['10Y']
                logger.info("Created BAA_10Y_spread as alternative")
                spread_col = 'BAA_10Y_spread'
                rating = 'BAA'
                tenor = '10Y'
        unique_regimes = regime_df['regime'].unique()
        if not hasattr(self, 'regime_params'):
            self.regime_params = {}
        for regime in unique_regimes:
            try:
                regime_dates = regime_df[regime_df['regime'] == regime].index
                regime_corp = corporate_df.loc[corporate_df.index.isin(regime_dates)]
                regime_yields = yields_df.loc[yields_df.index.isin(regime_dates)]
                logger.info(f"Calibrating {self.model_type} model to historical data for regime {regime}")
                self.calibrate(regime_corp, regime_yields, tenor=tenor, rating=rating)
                self.regime_params[regime] = self.params.copy()
                logger.info(f"Calibrated parameters for regime {regime}: {self.params}")
            except Exception as e:
                logger.error(f"Error calibrating for regime {regime}: {e}")
                self.regime_params[regime] = {
                    'asset_value': 1.2 - 0.05 * regime,
                    'debt_value': 1.0,
                    'asset_vol': 0.15 + 0.05 * regime,
                    'rating': rating,
                    'tenor': tenor
                }
                logger.info(f"Using fallback parameters for regime {regime}: {self.regime_params[regime]}")
    
    def simulate(self, T: float, dt: float, r_process: np.ndarray, regime: Optional[int] = None) -> np.ndarray:
        if regime is not None and regime in self.regime_params:
            params = self.regime_params[regime]
        else:
            params = self.params
        if not params:
            raise ValueError("Model not calibrated. Call calibrate() first.")
        if self.model_type == 'merton':
            return self.simulate_merton(
                asset_value=params['asset_value'],
                debt_value=params['debt_value'],
                asset_vol=params['asset_vol'],
                T=T,
                dt=dt,
                r_process=r_process
            )
        elif self.model_type == 'vasicek':
            steps = int(T / dt)
            spreads = np.zeros(steps + 1)
            spreads[0] = params.get('s0', 0.005)
            kappa = params['kappa']
            theta = params['theta']
            sigma = params['sigma']
            for t in range(1, steps + 1):
                dW = np.random.normal(0, np.sqrt(dt))
                ds = kappa * (theta - spreads[t-1]) * dt + sigma * dW
                spreads[t] = max(0, spreads[t-1] + ds)
            return spreads
        elif self.model_type == 'cir':
            steps = int(T / dt)
            spreads = np.zeros(steps + 1)
            spreads[0] = params.get('s0', 0.005)
            kappa = params['kappa']
            theta = params['theta']
            sigma = params['sigma']
            for t in range(1, steps + 1):
                dW = np.random.normal(0, np.sqrt(dt))
                ds = kappa * (theta - spreads[t-1]) * dt + sigma * np.sqrt(max(spreads[t-1], 0)) * dW
                spreads[t] = max(0, spreads[t-1] + ds)
            return spreads


class BondPricer:
    """
    Class for pricing fixed income instruments.
    """
    
    def __init__(self):
        pass
    
    def zero_coupon_bond_price(self, face_value: float, yield_rate: float, time_to_maturity: float) -> float:
        # FIX: Use continuous compounding discounting.
        return face_value * np.exp(-yield_rate * time_to_maturity)
    
    def simple_coupon_bond_price(self, face_value: float, coupon_rate: float, yield_rate: float, 
                                 time_to_maturity: float, payments_per_year: int = 2) -> float:
        """
        Calculate the price of a coupon-bearing bond using a realistic bond pricing formula.
        
        Args:
            face_value: Face value of the bond
            coupon_rate: Annual coupon rate (decimal)
            yield_rate: Annual yield rate (decimal)
            time_to_maturity: Time to maturity (years)
            payments_per_year: Number of coupon payments per year
            
        Returns:
            Bond price
        """
        # Handle extreme/invalid inputs
        if time_to_maturity <= 0:
            return face_value
        
        if yield_rate < 0:
            logger.warning(f"Negative yield rate {yield_rate} encountered, using 0.001 instead")
            yield_rate = 0.001
        
        # Calculate number of remaining payments
        n_payments = time_to_maturity * payments_per_year
        full_periods = int(n_payments)
        partial_period = n_payments - full_periods
        
        # Calculate coupon amount
        coupon_amount = face_value * coupon_rate / payments_per_year
        
        # Calculate price using standard bond pricing formula
        price = 0
        
        # Present value of coupon payments
        for i in range(1, full_periods + 1):
            t = (i - partial_period) / payments_per_year  # Time in years to this payment
            price += coupon_amount * np.exp(-yield_rate * t)
        
        # Present value of principal (face value)
        t_principal = (full_periods - partial_period) / payments_per_year
        price += face_value * np.exp(-yield_rate * t_principal)
        
        # Add accrued interest if in a partial period
        if partial_period > 0:
            accrued_interest = coupon_amount * partial_period
            price += accrued_interest
        
        return price
    
    def calculate_duration(self, face_value: float, coupon_rate: float, yield_rate: float, 
                           time_to_maturity: float, payments_per_year: int = 2) -> float:
        m = payments_per_year
        coupon_payment = face_value * coupon_rate / m
        n = int(np.ceil(time_to_maturity * m))
        times = np.array([(i+1)/m for i in range(n)])
        cash_flows = np.full(n, coupon_payment)
        cash_flows[-1] += face_value
        discount_factors = np.exp(-yield_rate * times)
        present_values = cash_flows * discount_factors
        price = np.sum(present_values)
        duration = np.sum(times * present_values) / price
        return duration
    
    def calculate_modified_duration(self, duration: float, yield_rate: float, payments_per_year: int = 2) -> float:
        period_yield = yield_rate / payments_per_year
        return duration / (1 + period_yield)
    
    def calculate_convexity(self, face_value: float, coupon_rate: float, yield_rate: float, 
                            time_to_maturity: float, payments_per_year: int = 2) -> float:
        m = payments_per_year
        coupon_payment = face_value * coupon_rate / m
        n = int(np.ceil(time_to_maturity * m))
        times = np.array([(i+1)/m for i in range(n)])
        cash_flows = np.full(n, coupon_payment)
        cash_flows[-1] += face_value
        discount_factors = np.exp(-yield_rate * times)
        present_values = cash_flows * discount_factors
        price = np.sum(present_values)
        convexity = np.sum((times**2) * present_values) / price
        return convexity
    
    def calculate_bond_yield(self, price: float, face_value: float, coupon_rate: float, 
                             time_to_maturity: float, payments_per_year: int = 2) -> float:
        def objective(ytm):
            calc_price = self.simple_coupon_bond_price(face_value, coupon_rate, ytm, time_to_maturity, payments_per_year)
            return (calc_price - price)**2
        initial_guess = coupon_rate
        result = minimize(objective, initial_guess, bounds=[(0.0001, 0.5)], method='L-BFGS-B')
        ytm = result.x[0]
        return ytm
    
    def bond_price_with_credit_risk(self, face_value: float, coupon_rate: float, 
                                    risk_free_rate: float, credit_spread: float, 
                                    time_to_maturity: float, previous_price: float = None,
                                    payments_per_year: int = 2) -> float:
        """
        Calculate the price of a bond with credit risk, using realistic bond math and
        with proper pull-to-par behavior.
        
        Args:
            face_value: Face value of the bond
            coupon_rate: Annual coupon rate (decimal)
            risk_free_rate: Annual risk-free rate (decimal)
            credit_spread: Annual credit spread (decimal)
            time_to_maturity: Time to maturity (years)
            previous_price: Previous day's price (for smoothing)
            payments_per_year: Number of coupon payments per year
            
        Returns:
            Bond price
        """
        # Handle extreme/invalid inputs
        if time_to_maturity <= 0:
            return face_value
        
        # Ensure positive yields and spreads
        risk_free_rate = max(0.001, risk_free_rate)
        credit_spread = max(0, credit_spread)
        
        # Calculate yield-to-maturity (risk-free rate + credit spread)
        yield_rate = risk_free_rate + credit_spread
        
        # Calculate theoretical bond price
        theoretical_price = self.simple_coupon_bond_price(
            face_value=face_value,
            coupon_rate=coupon_rate,
            yield_rate=yield_rate,
            time_to_maturity=time_to_maturity,
            payments_per_year=payments_per_year
        )
        
        # Enhanced pull-to-par mechanism
        # As bond approaches maturity, price should converge to face value
        # Increase the strength of pull-to-par as we get closer to maturity
        pull_to_par_strength = 0
        if time_to_maturity < 3.0:
            # Use a curve that accelerates pull-to-par as maturity approaches
            # Quadratic function gives smoother convergence to par value
            pull_to_par_strength = (1 - time_to_maturity/3.0)**2
            
            # Blend theoretical price with face value based on pull-to-par strength
            theoretical_price = (1 - pull_to_par_strength) * theoretical_price + pull_to_par_strength * face_value
        
        # If no previous price, return theoretical price
        if previous_price is None:
            return theoretical_price
        
        # Calculate price change percentage for adaptive smoothing
        price_change_pct = abs(theoretical_price - previous_price) / previous_price
        
        # Smoothing logic to prevent extreme daily price jumps
        # Reduce smoothing as we approach maturity to allow pull-to-par to work
        base_smoothing = 0.5  # 50% previous price, 50% theoretical
        
        # Reduce smoothing as maturity approaches
        if time_to_maturity < 1.0:
            # Linear reduction of smoothing weight
            adjusted_smoothing = base_smoothing * time_to_maturity
        else:
            adjusted_smoothing = base_smoothing
        
        # Only apply smoothing if price change is significant and we're not too close to maturity
        if price_change_pct > 0.01 and time_to_maturity > 0.25:
            # Apply smoothing
            smoothed_price = adjusted_smoothing * previous_price + (1 - adjusted_smoothing) * theoretical_price
            
            # Determine maximum daily price move
            # Base move amount (higher for higher duration bonds)
            estimated_duration = time_to_maturity if coupon_rate < 0.01 else min(time_to_maturity, 7.0)
            base_move = 0.01  # 1% base move
            max_daily_move = base_move * (1 + estimated_duration / 10) * previous_price
            
            # Adjust bounds based on time to maturity
            # Wider bounds as maturity approaches to allow price to converge to par
            maturity_factor = 1.0
            if time_to_maturity < 1.0:
                maturity_factor = 1.0 + (1.0 - time_to_maturity) * 3  # Up to 4x normal bounds near maturity
            
            # Apply bounds with maturity factor
            min_price = max(previous_price - max_daily_move * maturity_factor, 0.7 * face_value)
            max_price = min(previous_price + max_daily_move * maturity_factor, 1.3 * face_value)
            
            return np.clip(smoothed_price, min_price, max_price)
        else:
            # For small changes or near maturity, use theoretical price directly
            return theoretical_price


class BondMarketSimulator:
    """
    Class for simulating the bond market.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        self.interest_rate_model = InterestRateModel(random_state=random_state)
        self.credit_spread_model = CreditSpreadModel(random_state=random_state)
        self.bond_pricer = BondPricer()
        self.regime_detector = None
        self.regime_probabilities = None
        self.bond_universe = pd.DataFrame()
    
    def set_regime_detector(self, regime_detector):
        self.regime_detector = regime_detector
        if hasattr(regime_detector, 'get_regime_transitions'):
            self.regime_probabilities = regime_detector.get_regime_transitions(regime_detector.regime_df)
    
    def create_bond_universe(self, n_bonds: int = 100):
        logger.info(f"Creating realistic bond universe with {n_bonds} bonds")
        ratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC']
        rating_probs = [0.1, 0.15, 0.25, 0.25, 0.15, 0.07, 0.03]
        sectors = ['Treasury', 'Agency', 'Corporate-Financial', 'Corporate-Industrial', 'Corporate-Utility', 'Mortgage', 'Municipal', 'Emerging Markets']
        sector_probs = [0.2, 0.1, 0.15, 0.2, 0.1, 0.1, 0.1, 0.05]
        bonds = []
        for i in range(n_bonds):
            face_value = 1000.0
            rating = np.random.choice(ratings, p=rating_probs)
            if rating in ['AAA', 'AA', 'A', 'BBB']:
                coupon_rate = np.random.uniform(0.015, 0.05)
            else:
                coupon_rate = np.random.uniform(0.04, 0.08)
            maturity_bucket = np.random.choice(['Short', 'Medium', 'Long', 'Very Long'], p=[0.25, 0.35, 0.25, 0.15])
            if maturity_bucket == 'Short':
                time_to_maturity = np.random.uniform(1.0, 3.0)
            elif maturity_bucket == 'Medium':
                time_to_maturity = np.random.uniform(3.0, 7.0)
            elif maturity_bucket == 'Long':
                time_to_maturity = np.random.uniform(7.0, 15.0)
            else:
                time_to_maturity = np.random.uniform(15.0, 30.0)
            sector = np.random.choice(sectors, p=sector_probs)
            if rating == 'AAA':
                base_spread = np.random.uniform(0.001, 0.005)
            elif rating == 'AA':
                base_spread = np.random.uniform(0.004, 0.008)
            elif rating == 'A':
                base_spread = np.random.uniform(0.007, 0.015)
            elif rating == 'BBB':
                base_spread = np.random.uniform(0.014, 0.025)
            elif rating == 'BB':
                base_spread = np.random.uniform(0.022, 0.04)
            elif rating == 'B':
                base_spread = np.random.uniform(0.035, 0.065)
            else:
                base_spread = np.random.uniform(0.06, 0.12)
            if sector == 'Treasury':
                sector_adjustment = 0.0
            elif sector == 'Agency':
                sector_adjustment = 0.002
            elif 'Corporate' in sector:
                if 'Financial' in sector:
                    sector_adjustment = 0.004
                elif 'Industrial' in sector:
                    sector_adjustment = 0.003
                else:
                    sector_adjustment = 0.0035
            elif sector == 'Mortgage':
                sector_adjustment = 0.006
            elif sector == 'Municipal':
                sector_adjustment = 0.005
            else:
                sector_adjustment = 0.01
            maturity_adjustment = 0.001 * np.sqrt(time_to_maturity)
            credit_spread = base_spread + sector_adjustment + maturity_adjustment
            bond = {
                'id': f"BOND{i:04d}",
                'face_value': face_value,
                'coupon_rate': coupon_rate,
                'time_to_maturity': time_to_maturity,
                'rating': rating,
                'sector': sector,
                'credit_spread': credit_spread
            }
            bonds.append(bond)
        self.bond_universe = pd.DataFrame(bonds)
        logger.info(f"Created bond universe with {len(self.bond_universe)} bonds")
        return self.bond_universe
    
    def simulate_market(self, T: float = 1.0, dt: float = 1/252, 
                        risk_free_rate: float = 0.03, 
                        initial_regime: Optional[int] = None, 
                        include_bonds: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Simulate the bond market with realistic interest rate dynamics and bond price behavior.
        
        Args:
            T: Time horizon (years)
            dt: Time step (years)
            risk_free_rate: Initial risk-free rate
            initial_regime: Initial regime (if None, random regime will be selected)
            include_bonds: Whether to include bond pricing in the simulation
            
        Returns:
            Dictionary with simulated market data
        """
        logger.info(f"Simulating bond market for {T} years with dt={dt}")
        
        # Number of time steps
        steps = int(T / dt)
        
        # Initialize arrays with realistic time points
        times = np.linspace(0, T, steps + 1)
        
        # Simulate regime transitions
        regimes = np.zeros(steps + 1, dtype=int)
        
        # Determine initial regime
        if initial_regime is None:
            if self.regime_detector is not None and hasattr(self.regime_detector, 'regime_names'):
                if self.regime_detector.regime_names is not None:
                    regime_keys = list(self.regime_detector.regime_names.keys())
                    initial_regime = np.random.choice(regime_keys)
                else:
                    initial_regime = 0
                    logger.warning("regime_names is None, defaulting to regime 0")
            else:
                initial_regime = 0
        
        # Set initial regime
        regimes[0] = initial_regime
        
        # Simulate regime transitions if we have transition probabilities
        if self.regime_probabilities is not None:
            n_regimes = len(self.regime_probabilities.columns)
            for t in range(1, steps + 1):
                current_regime = regimes[t-1]
                # Get transition probabilities from current regime
                probs = self.regime_probabilities.iloc[current_regime].values
                # Ensure probabilities sum to 1 (fix any numerical issues)
                probs = probs / probs.sum()
                # Determine next regime
                regimes[t] = np.random.choice(range(n_regimes), p=probs)
        
        # Simulate interest rates with realistic dynamics
        short_rates = np.zeros(steps + 1)
        short_rates[0] = risk_free_rate  # Start with provided risk-free rate
        
        if self.interest_rate_model.params:
            # Model is calibrated, use it to simulate rates
            logger.info("Using calibrated interest rate model for simulation")
            
            # Determine which model to use based on model_type
            if self.interest_rate_model.model_type == 'vasicek':
                for t in range(1, steps + 1):
                    # Get current regime and its parameters
                    current_regime = regimes[t-1]
                    
                    # Get parameters for this regime
                    if current_regime in self.interest_rate_model.regime_params:
                        params = self.interest_rate_model.regime_params[current_regime]
                    else:
                        params = self.interest_rate_model.params
                    
                    # Extract parameters with fallbacks to reasonable values
                    kappa = params.get('kappa', 0.5)  # Mean reversion speed
                    theta = params.get('theta', 0.03)  # Long-term mean
                    sigma = params.get('sigma', 0.01)  # Volatility
                    
                    # Ensure parameters are in realistic ranges
                    kappa = min(max(kappa, 0.05), 5.0)
                    theta = min(max(theta, 0.005), 0.08)
                    sigma = min(max(sigma, 0.001), 0.05)
                    
                    # Simulate one step of Vasicek model
                    dW = np.random.normal(0, np.sqrt(dt))
                    dr = kappa * (theta - short_rates[t-1]) * dt + sigma * dW
                    
                    # Update rate with realistic bounds
                    short_rates[t] = min(max(short_rates[t-1] + dr, 0.001), 0.15)
            
            elif self.interest_rate_model.model_type == 'cir':
                for t in range(1, steps + 1):
                    # Get current regime and its parameters
                    current_regime = regimes[t-1]
                    
                    # Get parameters for this regime
                    if current_regime in self.interest_rate_model.regime_params:
                        params = self.interest_rate_model.regime_params[current_regime]
                    else:
                        params = self.interest_rate_model.params
                    
                    # Extract parameters with fallbacks to reasonable values
                    kappa = params.get('kappa', 0.5)  # Mean reversion speed
                    theta = params.get('theta', 0.03)  # Long-term mean
                    sigma = params.get('sigma', 0.01)  # Volatility
                    
                    # Ensure parameters are in realistic ranges
                    kappa = min(max(kappa, 0.05), 5.0)
                    theta = min(max(theta, 0.005), 0.08)
                    sigma = min(max(sigma, 0.001), 0.05)
                    
                    # Simulate one step of CIR model (ensures positive rates)
                    dW = np.random.normal(0, np.sqrt(dt))
                    dr = kappa * (theta - short_rates[t-1]) * dt + sigma * np.sqrt(max(0.001, short_rates[t-1])) * dW
                    
                    # Update rate with realistic bounds
                    short_rates[t] = min(max(short_rates[t-1] + dr, 0.001), 0.15)
        else:
            # Model not calibrated, use a robust mean-reverting process
            logger.warning("Interest rate model not calibrated. Using robust mean-reverting process.")
            
            # Use a more stable mean-reverting process with realistic parameters
            mean_level = risk_free_rate  # Long-term mean
            mean_reversion_speed = 0.3   # Speed of mean reversion (faster for stability)
            vol_factor = 0.2            # Volatility scaling factor
            
            for t in range(1, steps + 1):
                # Determine current regime
                current_regime = regimes[t-1]
                
                # Adjust parameters based on regime
                if current_regime == 0:  # Normal regime
                    regime_mean = mean_level
                    regime_vol = 0.004 * vol_factor
                elif current_regime == 1:  # Rising rate regime
                    regime_mean = mean_level * 1.2
                    regime_vol = 0.006 * vol_factor
                elif current_regime == 2:  # Falling rate regime
                    regime_mean = mean_level * 0.8
                    regime_vol = 0.005 * vol_factor
                else:  # Stressed regime
                    regime_mean = mean_level * 1.5
                    regime_vol = 0.008 * vol_factor
                
                # Simulate mean-reverting process
                mean_reversion = mean_reversion_speed * (regime_mean - short_rates[t-1]) * dt
                vol_term = regime_vol * np.random.normal(0, np.sqrt(dt))
                
                # Update rate with realistic bounds
                new_rate = short_rates[t-1] + mean_reversion + vol_term
                short_rates[t] = min(max(new_rate, 0.001), 0.10)
        
        # Simulate credit spreads with realistic dynamics
        credit_spreads = {rating: np.zeros(steps + 1) for rating in ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC']}
        
        # Set realistic base spreads by rating
        base_spreads = {
            'AAA': 0.001,   # 10 bps
            'AA': 0.0025,   # 25 bps
            'A': 0.005,     # 50 bps
            'BBB': 0.01,    # 100 bps
            'BB': 0.03,     # 300 bps
            'B': 0.045,     # 450 bps
            'CCC': 0.08     # 800 bps
        }
        
        # Define regime-specific spread parameters
        regime_spread_params = {
            # Normal regime
            0: {'drift_factor': -0.1, 'vol_factor': 0.8, 'mean_reversion_speed': 0.15},
            # Rising rate/widening spread regime
            1: {'drift_factor': 0.3, 'vol_factor': 1.5, 'mean_reversion_speed': 0.1},
            # Falling rate/tightening spread regime
            2: {'drift_factor': -0.4, 'vol_factor': 1.2, 'mean_reversion_speed': 0.2},
            # Stressed regime
            3: {'drift_factor': 0.6, 'vol_factor': 2.0, 'mean_reversion_speed': 0.05}
        }
        
        # Simulate credit spreads for each rating
        for rating in credit_spreads:
            # Set initial spread (use base spread for rating)
            spreads = np.zeros(steps + 1)
            spreads[0] = base_spreads[rating]
            
            # Scale factors for spread volatility based on rating quality
            rating_vol_factor = {
                'AAA': 0.4,
                'AA': 0.5,
                'A': 0.7,
                'BBB': 1.0,
                'BB': 1.4,
                'B': 1.8,
                'CCC': 2.5
            }[rating]
            
            # Simulate spread evolution
            for t in range(1, steps + 1):
                current_regime = regimes[t-1]
                
                # Get parameters for current regime
                params = regime_spread_params.get(current_regime, regime_spread_params[0])
                
                # Adjust parameters by rating
                drift = params['drift_factor'] * spreads[t-1] * dt
                vol = params['vol_factor'] * rating_vol_factor * 0.1 * spreads[t-1] * np.sqrt(dt)
                mean_reversion_speed = params['mean_reversion_speed']
                
                # Calculate baseline target spread (long-term mean)
                baseline_spread = base_spreads[rating] * (1 + 0.1 * current_regime)
                
                # Apply mean reversion
                mean_reversion = mean_reversion_speed * (baseline_spread - spreads[t-1]) * dt
                
                # Random shock
                dW = np.random.normal(0, 1)
                
                # Calculate spread change
                ds = mean_reversion + drift + vol * dW
                
                # Apply reasonable limits to daily changes
                max_daily_change = 0.1 * spreads[t-1]  # Limit to 10% change per day
                ds = np.clip(ds, -max_daily_change, max_daily_change)
                
                # Update spread with floor at small positive value
                spreads[t] = max(spreads[t-1] + ds, 0.0001)
                
                # Set maximum spread based on rating
                max_spread = base_spreads[rating] * 5
                spreads[t] = min(spreads[t], max_spread)
            
            # Store simulated spreads
            credit_spreads[rating] = spreads
        
        # Prepare bond price DataFrame
        bond_prices = pd.DataFrame()
        
        # Simulate bond prices if needed
        if include_bonds and not self.bond_universe.empty:
            bond_prices = pd.DataFrame(index=range(steps + 1))
            
            # Simulate prices for each bond
            for idx, bond in self.bond_universe.iterrows():
                # Initialize arrays for bond metrics
                prices = np.zeros(steps + 1)
                durations = np.zeros(steps + 1)
                convexities = np.zeros(steps + 1)
                
                # Get initial parameters
                face_value = bond['face_value']
                coupon_rate = bond['coupon_rate']
                initial_ttm = bond['time_to_maturity']
                
                # Calculate initial yield and price
                initial_rf_rate = short_rates[0]
                initial_spread = credit_spreads[bond['rating']][0]
                initial_yield = initial_rf_rate + initial_spread
                
                # Calculate initial theoretical price using accurate bond pricing
                initial_price = self.bond_pricer.bond_price_with_credit_risk(
                    face_value=face_value,
                    coupon_rate=coupon_rate,
                    risk_free_rate=initial_rf_rate,
                    credit_spread=initial_spread,
                    time_to_maturity=initial_ttm
                )
                
                # Apply reasonable bounds to initial price
                initial_price = np.clip(initial_price, 0.7 * face_value, 1.3 * face_value)
                
                # Set initial price
                prices[0] = initial_price
                
                # Log initial pricing info for a few bonds
                if idx < 5:
                    logger.info(f"Bond {bond['id']} initial price: ${initial_price:.2f}, coupon: {coupon_rate*100:.2f}%, yield: {initial_yield*100:.2f}%, ttm: {initial_ttm:.2f}y")
                
                # Calculate initial duration and convexity
                durations[0] = self.bond_pricer.calculate_duration(
                    face_value=face_value,
                    coupon_rate=coupon_rate,
                    yield_rate=initial_yield,
                    time_to_maturity=initial_ttm
                )
                
                convexities[0] = self.bond_pricer.calculate_convexity(
                    face_value=face_value,
                    coupon_rate=coupon_rate,
                    yield_rate=initial_yield,
                    time_to_maturity=initial_ttm
                )
                
                # Simulate prices for each time step
                for t in range(1, steps + 1):
                    # Calculate remaining time to maturity
                    time_to_maturity = max(0, initial_ttm - t * dt)
                    
                    # Pay face value at maturity
                    if time_to_maturity <= 0:
                        prices[t] = face_value
                        durations[t] = 0
                        convexities[t] = 0
                    else:
                        # Get current rates and spreads
                        rf_rate = short_rates[t]
                        spread = credit_spreads[bond['rating']][t]
                        
                        # Calculate theoretical price with credit risk
                        theoretical_price = self.bond_pricer.bond_price_with_credit_risk(
                            face_value=face_value,
                            coupon_rate=coupon_rate,
                            risk_free_rate=rf_rate,
                            credit_spread=spread,
                            time_to_maturity=time_to_maturity,
                            previous_price=prices[t-1]
                        )
                        
                        # Set price
                        prices[t] = theoretical_price
                        
                        # Calculate bond metrics based on new price
                        current_yield = rf_rate + spread
                        
                        # Calculate duration and convexity for risk measures
                        durations[t] = self.bond_pricer.calculate_duration(
                            face_value=face_value,
                            coupon_rate=coupon_rate,
                            yield_rate=current_yield,
                            time_to_maturity=time_to_maturity
                        )
                        
                        convexities[t] = self.bond_pricer.calculate_convexity(
                            face_value=face_value,
                            coupon_rate=coupon_rate,
                            yield_rate=current_yield,
                            time_to_maturity=time_to_maturity
                        )
                
                # Add bond data to DataFrame
                bond_prices[f"price_{bond['id']}"] = prices
                bond_prices[f"duration_{bond['id']}"] = durations
                bond_prices[f"convexity_{bond['id']}"] = convexities
        
        # Create rates DataFrame
        rates_df = pd.DataFrame({
            'time': times,
            'short_rate': short_rates,
            'regime': regimes
        })
        
        # Add credit spreads to rates DataFrame
        for rating, spreads in credit_spreads.items():
            rates_df[f"spread_{rating}"] = spreads
        
        logger.info(f"Simulated bond market with {len(rates_df)} time steps")
        
        return {
            'rates': rates_df,
            'bond_prices': bond_prices
        }
