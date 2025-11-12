#!/usr/bin/env python3
"""
Example: Running simulation with your own ETH price data

This script shows how to prepare and use real ETH data from exchanges
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from simulation import AMMSimulation
from data_utils import DataLoader
import config


def prepare_eth_data_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Load and prepare ETH data from CSV
    
    Expected CSV format:
        timestamp,open,high,low,close,volume
        2023-01-01 00:00:00,2000.5,2005.3,1998.2,2002.1,1000000
        ...
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Prepared DataFrame
    """
    print(f"Loading data from: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Parse timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    
    # Ensure we have required columns
    required_cols = ['close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Sort by time
    df = df.sort_index()
    
    # Remove any NaN values
    df = df.dropna(subset=['close'])
    
    print(f"Loaded {len(df)} data points")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    return df


def example_binance_api():
    """
    Example: Fetch ETH data from Binance API
    (Requires 'requests' library: pip install requests)
    """
    try:
        import requests
    except ImportError:
        print("Error: 'requests' library not installed")
        print("Install with: pip install requests")
        return None
    
    print("Fetching ETH/USDT data from Binance...")
    
    # Binance API endpoint
    url = "https://api.binance.com/api/v3/klines"
    
    # Parameters for 1-minute candles
    # Limit: 1000 candles per request (adjust as needed)
    params = {
        'symbol': 'ETHUSDT',
        'interval': '1m',
        'limit': 1000  # Last 1000 minutes (~16.7 hours)
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    # Parse data
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')
    
    # Convert string prices to float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    # Keep only necessary columns
    df = df[['open', 'high', 'low', 'close', 'volume']]
    
    print(f"Fetched {len(df)} candles from Binance")
    print(f"Latest price: ${df['close'].iloc[-1]:.2f}")
    
    return df


def run_simulation_with_real_data(csv_path: str = None,
                                  use_binance: bool = False):
    """
    Run simulation with real ETH data
    
    Args:
        csv_path: Path to local CSV file
        use_binance: Whether to fetch from Binance API
    """
    # Load data
    if use_binance:
        df = example_binance_api()
        if df is None:
            return
    elif csv_path:
        df = prepare_eth_data_from_csv(csv_path)
    else:
        print("Error: Must provide either csv_path or use_binance=True")
        return
    
    # Check if we have enough data
    min_samples = (config.TRAIN_HOURS + config.TEST_HOURS + 
                   config.VALIDATION_HOURS) * 60
    
    if len(df) < min_samples:
        print(f"\nWarning: Only {len(df)} samples available")
        print(f"Recommended: {min_samples} samples ({min_samples/60:.0f} hours)")
        print("Adjusting config to use available data...")
        
        # Adjust config
        available_hours = len(df) // 60
        config.TRAIN_HOURS = int(available_hours * 0.7)
        config.TEST_HOURS = int(available_hours * 0.2)
        config.VALIDATION_HOURS = int(available_hours * 0.1)
        
        print(f"New split: Train={config.TRAIN_HOURS}h, "
              f"Test={config.TEST_HOURS}h, Val={config.VALIDATION_HOURS}h")
    
    # Split data
    loader = DataLoader()
    loader.data = df
    train_df, test_df, val_df = loader.split_data(df)
    
    train_prices = train_df['close'].values
    test_prices = test_df['close'].values
    
    initial_price = train_prices[0]
    
    print(f"\n{'='*70}")
    print("Running AMM Simulation with Real ETH Data")
    print(f"{'='*70}")
    print(f"Initial Price: ${initial_price:.2f}")
    print(f"Liquidity Constant: {config.LIQUIDITY_CONSTANT:,}")
    print(f"Training on {len(train_prices)} samples ({len(train_prices)/60:.1f} hours)")
    print(f"Testing on {len(test_prices)} samples ({len(test_prices)/60:.1f} hours)")
    print(f"{'='*70}\n")
    
    # Create and run simulation
    simulation = AMMSimulation(initial_price, config.LIQUIDITY_CONSTANT)
    results = simulation.run_simulation(train_prices, test_prices)
    
    # Save outputs
    simulation.save_models()
    simulation.plot_results()
    
    print("\n✅ Simulation completed successfully!")
    
    return results


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║        Real ETH Data Simulation Example                       ║
    ╚════════════════════════════════════════════════════════════════╝
    
    Choose data source:
    
    1. Load from local CSV file
       - Prepare CSV with: timestamp, open, high, low, close, volume
       - 1-minute interval data recommended
    
    2. Fetch from Binance API
       - Gets latest 1000 1-minute candles (~16.7 hours)
       - Requires 'requests' library
    
    3. Cancel
    """)
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == '1':
        csv_path = input("\nEnter path to CSV file: ").strip()
        if os.path.exists(csv_path):
            run_simulation_with_real_data(csv_path=csv_path)
        else:
            print(f"Error: File not found: {csv_path}")
    
    elif choice == '2':
        run_simulation_with_real_data(use_binance=True)
    
    else:
        print("Cancelled.")
