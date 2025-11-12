"""
Data Preparation Module
Utilities for loading and preprocessing ETH price data
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import config


class DataLoader:
    """Load and preprocess price data"""
    
    def __init__(self, filepath: str = None):
        """
        Initialize data loader
        
        Args:
            filepath: Path to price data CSV file
        """
        self.filepath = filepath
        self.data = None
        
    def load_csv(self, filepath: str = None) -> pd.DataFrame:
        """
        Load price data from CSV
        
        Expected columns: timestamp, open, high, low, close, volume
        
        Timestamp formats supported:
        - Unix timestamp (seconds): 1502901900
        - Datetime string: "2023-01-01 00:00:00"
        - Datetime string: "2023-01-01T00:00:00"
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with price data
        """
        if filepath is None:
            filepath = self.filepath
        
        df = pd.read_csv(filepath)
        
        # Ensure timestamp column
        if 'timestamp' in df.columns:
            # Check if Unix timestamp (integer) or datetime string
            if df['timestamp'].dtype in ['int64', 'int32', 'float64']:
                # Unix timestamp - convert from seconds
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            else:
                # Datetime string - parse automatically
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            df = df.set_index('timestamp')
        
        # Validate required columns
        required = ['close']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        self.data = df
        return df
    
    def generate_synthetic_data(self, n_hours: int = 10000,
                               initial_price: float = 2000,
                               volatility: float = 0.02,
                               drift: float = 0.0001) -> pd.DataFrame:
        """
        Generate synthetic price data using geometric Brownian motion
        Useful for testing when real data is not available
        
        Args:
            n_hours: Number of hours to simulate
            initial_price: Starting price
            volatility: Price volatility (σ)
            drift: Price drift (μ)
            
        Returns:
            DataFrame with synthetic price data
        """
        np.random.seed(config.RANDOM_SEED)
        
        # Generate 1-minute intervals
        n_minutes = n_hours * 60
        
        # Geometric Brownian motion: dS = μS dt + σS dW
        dt = 1 / (60 * 24 * 365)  # 1 minute in years
        
        prices = [initial_price]
        for _ in range(n_minutes - 1):
            dW = np.random.normal(0, np.sqrt(dt))
            dS = drift * prices[-1] * dt + volatility * prices[-1] * dW
            new_price = prices[-1] + dS
            prices.append(max(new_price, 0.01))  # Ensure positive
        
        # Create DataFrame
        timestamps = pd.date_range(
            start='2023-01-01',
            periods=n_minutes,
            freq='1min'
        )
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'close': prices,
            'open': prices,  # Simplified
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'volume': np.random.lognormal(10, 1, n_minutes)
        })
        
        df = df.set_index('timestamp')
        self.data = df
        
        return df
    
    def split_data(self, df: pd.DataFrame = None,
                  train_hours: int = None,
                  test_hours: int = None,
                  validation_hours: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, test, and validation sets
        
        Args:
            df: DataFrame to split
            train_hours: Hours for training
            test_hours: Hours for testing
            validation_hours: Hours for validation
            
        Returns:
            (train_df, test_df, validation_df)
        """
        if df is None:
            df = self.data
        
        if df is None:
            raise ValueError("No data loaded")
        
        train_hours = train_hours or config.TRAIN_HOURS
        test_hours = test_hours or config.TEST_HOURS
        validation_hours = validation_hours or config.VALIDATION_HOURS
        
        # Convert hours to number of 1-minute intervals
        train_samples = train_hours * 60
        test_samples = test_hours * 60
        validation_samples = validation_hours * 60
        
        # Split
        train_df = df.iloc[:train_samples]
        test_df = df.iloc[train_samples:train_samples + test_samples]
        validation_df = df.iloc[train_samples + test_samples:
                                train_samples + test_samples + validation_samples]
        
        return train_df, test_df, validation_df
    
    def calculate_returns(self, df: pd.DataFrame = None) -> np.ndarray:
        """
        Calculate log returns
        
        Args:
            df: DataFrame with prices
            
        Returns:
            Array of log returns
        """
        if df is None:
            df = self.data
        
        prices = df['close'].values
        returns = np.diff(np.log(prices))
        
        return returns
    
    def detect_price_events(self, df: pd.DataFrame = None,
                           threshold: float = None) -> np.ndarray:
        """
        Detect significant price change events
        Following Algorithm 1 from the paper
        
        Args:
            df: DataFrame with prices
            threshold: β_v threshold for price changes
            
        Returns:
            Boolean array indicating event occurrences
        """
        if df is None:
            df = self.data
        
        threshold = threshold or config.BETA_V
        
        prices = df['close'].values
        
        # Calculate price changes
        price_changes = np.diff(prices) / prices[:-1]
        
        # Detect events: |price_change| > threshold
        events = np.abs(price_changes) > threshold
        
        # Pad to match original length
        events = np.concatenate([[False], events])
        
        return events


class TradeGenerator:
    """Generate synthetic trades for simulation"""
    
    def __init__(self, amm_liquidity: float):
        """
        Initialize trade generator
        
        Args:
            amm_liquidity: Total AMM liquidity (sqrt of c)
        """
        self.amm_liquidity = amm_liquidity
        
    def generate_trades(self, n_intervals: int,
                       trade_frequency: float = None,
                       min_size_pct: float = None,
                       max_size_pct: float = None) -> list:
        """
        Generate synthetic trades
        
        Args:
            n_intervals: Number of time intervals
            trade_frequency: Probability of trade per interval
            min_size_pct: Minimum trade size as % of liquidity
            max_size_pct: Maximum trade size as % of liquidity
            
        Returns:
            List of (interval_idx, size, is_buy) tuples
        """
        trade_frequency = trade_frequency or config.TRADE_FREQUENCY
        min_size_pct = min_size_pct or config.MIN_TRADE_SIZE
        max_size_pct = max_size_pct or config.MAX_TRADE_SIZE
        
        np.random.seed(config.RANDOM_SEED)
        
        trades = []
        
        for i in range(n_intervals):
            # Decide if trade occurs
            if np.random.random() < trade_frequency:
                # Generate trade size
                size_pct = np.random.uniform(min_size_pct, max_size_pct)
                size = self.amm_liquidity * size_pct / 100
                
                # Decide direction (buy or sell)
                is_buy = np.random.random() > 0.5
                
                trades.append((i, size, is_buy))
        
        return trades
    
    def generate_market_impact_trades(self, prices: np.ndarray,
                                     volatility_factor: float = 1.0) -> list:
        """
        Generate trades that correlate with price movements
        More realistic than purely random trades
        
        Args:
            prices: Price series
            volatility_factor: Factor to scale trade sizes
            
        Returns:
            List of trades
        """
        n_intervals = len(prices)
        trades = []
        
        # Calculate price changes
        price_changes = np.diff(prices) / prices[:-1]
        
        for i in range(len(price_changes)):
            # Higher probability of trade during volatile periods
            volatility = abs(price_changes[i])
            trade_prob = min(0.5, volatility * 10)
            
            if np.random.random() < trade_prob:
                # Trade size correlates with price change magnitude
                size_pct = min(
                    config.MAX_TRADE_SIZE,
                    config.MIN_TRADE_SIZE + abs(price_changes[i]) * 100 * volatility_factor
                )
                size = self.amm_liquidity * size_pct / 100
                
                # Direction: buy if price increasing, sell if decreasing
                # Add some randomness
                is_buy = (price_changes[i] > 0) if np.random.random() > 0.3 else (price_changes[i] < 0)
                
                trades.append((i, size, is_buy))
        
        return trades


def load_eth_data(filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load ETH data and split into train/test/val
    
    Args:
        filepath: Path to CSV file with ETH data
        
    Returns:
        (train_df, test_df, val_df)
    """
    loader = DataLoader(filepath)
    df = loader.load_csv()
    return loader.split_data(df)


def create_synthetic_eth_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create synthetic ETH-like data for testing
    
    Returns:
        (train_df, test_df, val_df)
    """
    loader = DataLoader()
    
    total_hours = config.TRAIN_HOURS + config.TEST_HOURS + config.VALIDATION_HOURS
    
    df = loader.generate_synthetic_data(
        n_hours=total_hours,
        initial_price=2000,  # Realistic ETH price
        volatility=0.03,  # ETH-like volatility
        drift=0.0001
    )
    
    return loader.split_data(df)
