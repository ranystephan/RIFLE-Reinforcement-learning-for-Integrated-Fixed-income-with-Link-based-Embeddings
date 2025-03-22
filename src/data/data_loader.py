"""
Data Loader Module for Fixed Income RL Project

This module contains functions to fetch various fixed income data from public sources:
1. US Treasury Yield Curve data from FRED
2. Corporate bond indices from FRED/Quandl
3. Macroeconomic indicators from FRED
4. Individual bond data

Author: ranycs & cosrv
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fredapi import Fred
import quandl
import yfinance as yf
from typing import List, Dict, Union, Tuple, Optional
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FixedIncomeDataLoader:
    """
    Class for loading fixed income data from various sources.
    """
    
    def __init__(self, fred_api_key: str = None, quandl_api_key: str = None):
        """
        Initialize the data loader with API keys.
        
        Args:
            fred_api_key: API key for FRED (Federal Reserve Economic Data)
            quandl_api_key: API key for Quandl
        """
        self.fred_api_key = fred_api_key
        self.quandl_api_key = quandl_api_key
        
        # Initialize API clients
        if fred_api_key:
            self.fred = Fred(api_key=fred_api_key)
        else:
            logger.warning("FRED API key not provided. Some data may not be accessible.")
            self.fred = None
            
        if quandl_api_key:
            quandl.ApiConfig.api_key = quandl_api_key
        else:
            logger.warning("Quandl API key not provided. Some data may not be accessible.")
    
    def get_treasury_yields(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch US Treasury yield data from FRED.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with daily yield curve data
        """
        logger.info(f"Fetching Treasury yields from {start_date} to {end_date}")
        
        # Treasury constant maturity rate series IDs
        treasury_series = {
            '1M': 'DGS1MO',
            '3M': 'DGS3MO',
            '6M': 'DGS6MO',
            '1Y': 'DGS1',
            '2Y': 'DGS2',
            '3Y': 'DGS3',
            '5Y': 'DGS5',
            '7Y': 'DGS7',
            '10Y': 'DGS10',
            '20Y': 'DGS20',
            '30Y': 'DGS30'
        }
        
        if not self.fred:
            raise ValueError("FRED API key is required for fetching Treasury yields")
        
        # Fetch data for each maturity
        yield_data = {}
        for maturity, series_id in treasury_series.items():
            try:
                series = self.fred.get_series(
                    series_id, 
                    observation_start=start_date,
                    observation_end=end_date
                )
                yield_data[maturity] = series
            except Exception as e:
                logger.error(f"Error fetching {maturity} Treasury yield: {e}")
        
        # Combine into a single DataFrame
        yields_df = pd.DataFrame(yield_data)
        
        # Forward fill missing values (weekends/holidays)
        yields_df = yields_df.ffill()
        
        # Calculate key curve metrics
        if '10Y' in yields_df.columns and '2Y' in yields_df.columns:
            yields_df['2s10s_spread'] = yields_df['10Y'] - yields_df['2Y']
        
        if '30Y' in yields_df.columns and '3M' in yields_df.columns:
            yields_df['3m30y_spread'] = yields_df['30Y'] - yields_df['3M']
        
        # Calculate level, slope, curvature (classic yield curve factors)
        if all(x in yields_df.columns for x in ['2Y', '5Y', '10Y']):
            yields_df['level'] = (yields_df['2Y'] + yields_df['5Y'] + yields_df['10Y']) / 3
            yields_df['slope'] = yields_df['10Y'] - yields_df['2Y']
            yields_df['curvature'] = 2 * yields_df['5Y'] - yields_df['2Y'] - yields_df['10Y']
        
        logger.info(f"Successfully fetched yield data with shape {yields_df.shape}")
        return yields_df
    
    def get_corporate_bond_indices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch corporate bond indices from FRED.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with corporate bond indices
        """
        logger.info(f"Fetching corporate bond indices from {start_date} to {end_date}")
        
        # Corporate bond index series IDs
        corp_series = {
            'AAA': 'DAAA',          # Moody's Seasoned Aaa Corporate Bond Yield
            'BAA': 'DBAA',          # Moody's Seasoned Baa Corporate Bond Yield
            'AAA_10Y_spread': 'AAA10Y',  # AAA Corporate Bond Spread (AAA - 10Y Treasury)
            'BAA_10Y_spread': 'BAA10Y',  # BAA Corporate Bond Spread (BAA - 10Y Treasury)
            'High_Yield_Index': 'BAMLH0A0HYM2'  # ICE BofA US High Yield Index
        }
        
        if not self.fred:
            raise ValueError("FRED API key is required for fetching corporate bond indices")
        
        # Fetch data for each index
        corp_data = {}
        for index_name, series_id in corp_series.items():
            try:
                series = self.fred.get_series(
                    series_id, 
                    observation_start=start_date,
                    observation_end=end_date
                )
                corp_data[index_name] = series
            except Exception as e:
                logger.error(f"Error fetching {index_name} corporate bond index: {e}")
        
        # Combine into a single DataFrame
        corp_df = pd.DataFrame(corp_data)
        
        # Forward fill missing values (weekends/holidays)
        corp_df = corp_df.ffill()
        
        # Calculate additional metrics
        if 'AAA' in corp_df.columns and 'BAA' in corp_df.columns:
            corp_df['AAA_BAA_spread'] = corp_df['BAA'] - corp_df['AAA']
        
        logger.info(f"Successfully fetched corporate bond indices with shape {corp_df.shape}")
        return corp_df
    
    def get_macro_indicators(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch macroeconomic indicators from FRED.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with macroeconomic indicators
        """
        logger.info(f"Fetching macroeconomic indicators from {start_date} to {end_date}")
        
        # Macroeconomic indicator series IDs
        macro_series = {
            'CPI': 'CPIAUCSL',          # Consumer Price Index for All Urban Consumers
            'Core_CPI': 'CPILFESL',     # Core CPI (All Items Less Food and Energy)
            'Unemployment': 'UNRATE',    # Unemployment Rate
            'GDP': 'GDPC1',              # Real Gross Domestic Product
            'Industrial_Production': 'INDPRO',  # Industrial Production Index
            'Retail_Sales': 'RSAFS',     # Retail Sales
            'Housing_Starts': 'HOUST',   # Housing Starts
            'Consumer_Sentiment': 'UMCSENT',  # University of Michigan Consumer Sentiment
            'VIX': 'VIXCLS'             # CBOE Volatility Index
        }
        
        if not self.fred:
            raise ValueError("FRED API key is required for fetching macroeconomic indicators")
        
        # Fetch data for each indicator
        macro_data = {}
        for indicator, series_id in macro_series.items():
            try:
                series = self.fred.get_series(
                    series_id, 
                    observation_start=start_date,
                    observation_end=end_date
                )
                macro_data[indicator] = series
            except Exception as e:
                logger.error(f"Error fetching {indicator}: {e}")
        
        # Combine into a single DataFrame
        macro_df = pd.DataFrame(macro_data)
        
        # Calculate inflation metrics (YoY % change)
        if 'CPI' in macro_df.columns:
            macro_df['Inflation_YoY'] = macro_df['CPI'].pct_change(periods=12) * 100
            
        if 'Core_CPI' in macro_df.columns:
            macro_df['Core_Inflation_YoY'] = macro_df['Core_CPI'].pct_change(periods=12) * 100
        
        # Forward fill missing values
        macro_df = macro_df.ffill()
        
        logger.info(f"Successfully fetched macro indicators with shape {macro_df.shape}")
        return macro_df
    
    def get_etf_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch bond ETF data as a proxy for sectoral bond performance.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with bond ETF data
        """
        logger.info(f"Fetching bond ETF data from {start_date} to {end_date}")
        
        # Bond ETF tickers by category
        bond_etfs = {
            'Treasury': ['IEF', 'TLT', 'SHY'],  # Intermediate, Long-term, Short-term Treasury
            'Corporate': ['LQD', 'VCIT', 'VCSH'],  # Investment Grade, Int. Corp, Short Corp
            'High_Yield': ['HYG', 'JNK'],  # High Yield Corporate
            'Municipal': ['MUB', 'TFI'],  # Municipal Bonds
            'International': ['BNDX', 'EMB']  # International and Emerging Markets
        }
        
        # Flatten the dictionary to get all tickers
        all_tickers = [ticker for category in bond_etfs.values() for ticker in category]
        
        # Create empty DataFrames to store results
        price_data = pd.DataFrame()
        returns_data = pd.DataFrame()
        volatility_data = pd.DataFrame()
        
        # Fetch data for each ticker individually to avoid connection pool issues
        successful_tickers = []
        for ticker in all_tickers:
            try:
                # Add a longer timeout and retry mechanism
                for attempt in range(3):  # Try up to 3 times
                    try:
                        # Download data for single ticker
                        ticker_data = yf.download(
                            ticker, 
                            start=start_date, 
                            end=end_date,
                            progress=False,
                            timeout=30  # Increase timeout to 30 seconds
                        )
                        
                        if not ticker_data.empty:
                            # Try to get price data from different columns in order of preference
                            for col in ['Adj Close', 'Close', 'Price']:
                                if col in ticker_data.columns:
                                    price_series = ticker_data[col]
                                    logger.info(f"Using {col} prices for {ticker}")
                                    
                                    # Store price data
                                    price_data[f'price_{ticker}'] = price_series
                                    
                                    # Calculate returns
                                    returns = price_series.pct_change().fillna(0)
                                    returns_data[f'return_{ticker}'] = returns
                                    
                                    # Calculate volatility
                                    volatility = returns.rolling(window=20).std().fillna(0) * np.sqrt(252)
                                    volatility_data[f'vol_{ticker}'] = volatility
                                    
                                    successful_tickers.append(ticker)
                                    break
                            else:
                                logger.warning(f"No suitable price column found for {ticker}")
                                continue
                            
                            break  # Break retry loop if successful
                            
                    except Exception as e:
                        if attempt < 2:  # If not the last attempt
                            logger.warning(f"Attempt {attempt+1} failed for {ticker}: {e}. Retrying...")
                            time.sleep(2)  # Wait before retrying
                        else:
                            raise e
                            
            except Exception as e:
                logger.warning(f"Error fetching data for {ticker}: {e}")
                # Continue with next ticker
                continue
        
        if not successful_tickers:
            logger.warning("No ETF data could be fetched successfully")
            # Create empty DataFrame with proper structure
            logger.info("Creating empty ETF data structure")
            empty_df = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq='B'))
            
            # Create columns for each expected ticker and metric
            for ticker in all_tickers:
                empty_df[f'price_{ticker}'] = np.nan
                empty_df[f'return_{ticker}'] = 0.0
                empty_df[f'vol_{ticker}'] = 0.0
            
            return empty_df
        
        # Combine all data
        result = pd.concat([price_data, returns_data, volatility_data], axis=1)
        
        # Forward fill any missing values
        result = result.ffill()
        
        logger.info(f"Successfully fetched ETF data for {len(successful_tickers)} tickers with shape {result.shape}")
        logger.info(f"Successfully loaded tickers: {', '.join(successful_tickers)}")
        return result
    
    def get_bond_issue_relationships(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create a dataset representing relationships between bond issuers.
        This is a simplified version that uses sector/industry relationships.
        
        Returns:
            Tuple of (nodes_df, edges_df) for graph construction
        """
        logger.info("Generating bond issuer relationship data")
        
        # For a real project, we would use actual bond issue data
        # Here we'll create a simplified representation using S&P 500 companies
        try:
            # Get S&P 500 companies as a proxy for bond issuers
            sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            sp500 = sp500.rename(columns={'Symbol': 'ticker', 'Security': 'name', 
                                        'GICS Sector': 'sector', 'GICS Sub-Industry': 'industry'})
            
            # Create nodes DataFrame (issuers)
            nodes_df = sp500[['ticker', 'name', 'sector', 'industry']].copy()
            
            # Set ticker as the index - ensure it's string type for consistent mapping
            nodes_df['ticker'] = nodes_df['ticker'].astype(str)
            nodes_df = nodes_df.set_index('ticker')
            
            # Add a simulated credit rating
            np.random.seed(42)  # For reproducibility
            ratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B']
            weights = [0.05, 0.15, 0.3, 0.3, 0.15, 0.05]  # Probability distribution
            nodes_df['rating'] = np.random.choice(ratings, size=len(nodes_df), p=weights)
            
            # Create edges DataFrame (relationships between issuers)
            edges = []
            
            # Companies in the same sector have relationships
            for sector in nodes_df['sector'].unique():
                # Get companies in this sector - use index values (tickers)
                sector_companies = nodes_df[nodes_df['sector'] == sector].index.tolist()
                
                # Companies in the same industry have stronger relationships
                for industry in nodes_df[nodes_df['sector'] == sector]['industry'].unique():
                    # Get companies in this industry - use index values (tickers)
                    industry_companies = nodes_df[nodes_df['industry'] == industry].index.tolist()
                    
                    # Create edges between companies in the same industry
                    for i in range(len(industry_companies)):
                        for j in range(i+1, len(industry_companies)):
                            edges.append({
                                'source': industry_companies[i],
                                'target': industry_companies[j],
                                'relationship': 'industry',
                                'weight': 0.7  # Stronger relationship
                            })
                
                # Create some edges between companies in the same sector but different industries
                for i in range(len(sector_companies)):
                    for j in range(i+1, len(sector_companies)):
                        source = sector_companies[i]
                        target = sector_companies[j]
                        
                        # Only add if they're not in the same industry
                        if nodes_df.loc[source, 'industry'] != nodes_df.loc[target, 'industry']:
                            # Add with some probability to avoid a fully connected graph
                            if np.random.random() < 0.1:
                                edges.append({
                                    'source': source,
                                    'target': target,
                                    'relationship': 'sector',
                                    'weight': 0.3  # Weaker relationship
                                })
            
            edges_df = pd.DataFrame(edges)
            
            # Ensure source and target are string type for consistent mapping
            edges_df['source'] = edges_df['source'].astype(str)
            edges_df['target'] = edges_df['target'].astype(str)
            
            # Verify that all edge nodes exist in the nodes dataframe
            source_exists = edges_df['source'].isin(nodes_df.index)
            target_exists = edges_df['target'].isin(nodes_df.index)
            
            # Filter out edges with invalid nodes
            valid_edges = edges_df[source_exists & target_exists]
            if len(valid_edges) < len(edges_df):
                logger.warning(f"Filtered out {len(edges_df) - len(valid_edges)} edges with invalid node references")
                edges_df = valid_edges
            
            logger.info(f"Successfully generated relationship data with {len(nodes_df)} nodes and {len(edges_df)} edges")
            logger.info(f"Node index type: {type(nodes_df.index[0])}")
            logger.info(f"Edge source type: {type(edges_df['source'].iloc[0]) if not edges_df.empty else 'N/A'}")
            
            return nodes_df, edges_df
            
        except Exception as e:
            logger.error(f"Error generating relationship data: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def save_to_csv(self, data: pd.DataFrame, filename: str, directory: str = 'data/raw'):
        """
        Save the data to a CSV file.
        
        Args:
            data: DataFrame to save
            filename: Name of the file
            directory: Directory to save the file in
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        filepath = os.path.join(directory, filename)
        data.to_csv(filepath)
        logger.info(f"Saved data to {filepath}")
    
    def load_all_data(self, start_date: str, end_date: str, save: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Load all necessary data for the fixed income RL project.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            save: Whether to save the data to CSV files
            
        Returns:
            Dictionary with all loaded data
        """
        logger.info(f"Loading all data from {start_date} to {end_date}")
        
        data = {}
        
        # Load Treasury yields
        try:
            data['yields'] = self.get_treasury_yields(start_date, end_date)
            if save:
                self.save_to_csv(data['yields'], 'treasury_yields.csv')
        except Exception as e:
            logger.error(f"Error loading Treasury yields: {e}")
        
        # Load corporate bond indices
        try:
            data['corporate'] = self.get_corporate_bond_indices(start_date, end_date)
            if save:
                self.save_to_csv(data['corporate'], 'corporate_bonds.csv')
        except Exception as e:
            logger.error(f"Error loading corporate bond indices: {e}")
        
        # Load macroeconomic indicators
        try:
            data['macro'] = self.get_macro_indicators(start_date, end_date)
            if save:
                self.save_to_csv(data['macro'], 'macro_indicators.csv')
        except Exception as e:
            logger.error(f"Error loading macroeconomic indicators: {e}")
        
        # Load ETF data
        try:
            data['etfs'] = self.get_etf_data(start_date, end_date)
            if save:
                self.save_to_csv(data['etfs'], 'bond_etfs.csv')
        except Exception as e:
            logger.error(f"Error loading ETF data: {e}")
        
        # Load bond issue relationships
        try:
            nodes_df, edges_df = self.get_bond_issue_relationships()
            data['issuer_nodes'] = nodes_df
            data['issuer_edges'] = edges_df
            if save:
                self.save_to_csv(nodes_df, 'issuer_nodes.csv')
                self.save_to_csv(edges_df, 'issuer_edges.csv')
        except Exception as e:
            logger.error(f"Error loading bond issue relationships: {e}")
        
        return data
