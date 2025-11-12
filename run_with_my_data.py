#!/usr/bin/env python3
"""
Verify and run simulation with your ETH CSV data
Handles Unix timestamp format automatically
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from simulation import run_eth_simulation
from data_utils import DataLoader
import config


def verify_csv_format(csv_path: str):
    """
    Verify CSV format and show data info
    
    Args:
        csv_path: Path to your CSV file
    """
    print(f"\n{'='*70}")
    print("📊 Verifying CSV Format")
    print(f"{'='*70}\n")
    
    print(f"Loading: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    print(f"\n✅ CSV loaded successfully!")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")
    
    # Check required columns
    required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        print(f"\n❌ Missing columns: {missing}")
        return False
    
    print(f"\n✅ All required columns present")
    
    # Show sample data
    print(f"\n📋 First 3 rows:")
    print(df.head(3).to_string())
    
    # Parse timestamp
    print(f"\n🕐 Parsing timestamps...")
    
    # Handle Unix timestamp (your format)
    if df['timestamp'].dtype in ['int64', 'int32']:
        print(f"   Detected: Unix timestamp format")
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    else:
        print(f"   Detected: String datetime format")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df = df.set_index('timestamp')
    
    print(f"\n✅ Timestamps parsed successfully!")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    print(f"   Duration: {(df.index[-1] - df.index[0]).days} days")
    
    # Calculate time interval
    time_diffs = df.index.to_series().diff().dropna()
    median_interval = time_diffs.median()
    print(f"   Median interval: {median_interval.total_seconds()} seconds")
    
    if median_interval.total_seconds() == 60:
        print(f"   ✅ Perfect! 1-minute intervals as expected")
    else:
        print(f"   ⚠️  Note: Expected 60s, got {median_interval.total_seconds()}s")
    
    # Price statistics
    print(f"\n💰 Price Statistics:")
    print(f"   Mean:   ${df['close'].mean():.2f}")
    print(f"   Min:    ${df['close'].min():.2f}")
    print(f"   Max:    ${df['close'].max():.2f}")
    print(f"   Std:    ${df['close'].std():.2f}")
    
    # Check for data quality issues
    print(f"\n🔍 Data Quality Checks:")
    
    # Missing values
    missing_close = df['close'].isna().sum()
    if missing_close > 0:
        print(f"   ⚠️  Missing close prices: {missing_close}")
    else:
        print(f"   ✅ No missing close prices")
    
    # Zero prices
    zero_prices = (df['close'] == 0).sum()
    if zero_prices > 0:
        print(f"   ⚠️  Zero prices found: {zero_prices}")
    else:
        print(f"   ✅ No zero prices")
    
    # Duplicate timestamps
    duplicates = df.index.duplicated().sum()
    if duplicates > 0:
        print(f"   ⚠️  Duplicate timestamps: {duplicates}")
    else:
        print(f"   ✅ No duplicate timestamps")
    
    # Data sufficiency
    hours_available = len(df) / 60
    hours_needed = config.TRAIN_HOURS + config.TEST_HOURS + config.VALIDATION_HOURS
    
    print(f"\n📊 Data Sufficiency:")
    print(f"   Available: {hours_available:.1f} hours")
    print(f"   Needed:    {hours_needed:.1f} hours")
    
    if hours_available >= hours_needed:
        print(f"   ✅ Sufficient data!")
    else:
        print(f"   ⚠️  Less data than recommended")
        print(f"      Simulation will auto-adjust to use available data")
    
    print(f"\n{'='*70}")
    print("✅ CSV format verified and ready to use!")
    print(f"{'='*70}\n")
    
    return True


def run_with_your_data(csv_path: str):
    """
    Run simulation with your CSV data
    
    Args:
        csv_path: Path to your CSV file
    """
    # First verify the format
    if not verify_csv_format(csv_path):
        print("\n❌ CSV verification failed. Please check your data format.")
        return
    
    # Ask for confirmation
    print("\nReady to run simulation with your data.")
    print("\nThis will:")
    print("  1. Load your ETH price data")
    print("  2. Train LSTM model (~5-10 minutes)")
    print("  3. Train Q-learning agent (~5-10 minutes)")
    print("  4. Compare proposed AMM vs baseline")
    print("  5. Generate plots and metrics")
    
    response = input("\nContinue? (y/n): ").strip().lower()
    
    if response != 'y':
        print("\nCancelled.")
        return
    
    print(f"\n{'='*70}")
    print("🚀 Starting AMM Simulation")
    print(f"{'='*70}\n")
    
    # Run simulation
    try:
        results = run_eth_simulation(
            eth_data_path=csv_path,
            use_synthetic=False
        )
        
        print(f"\n{'='*70}")
        print("✅ SUCCESS! Simulation completed.")
        print(f"{'='*70}")
        
        print(f"\nResults Summary:")
        print(f"  📉 Divergence Loss Reduction: {results['divergence_loss']['improvement']:.2f}%")
        print(f"  📉 Slippage Reduction: {results['slippage']['improvement']:.2f}%")
        print(f"  📈 Liquidity Utilization: {results['liquidity_utilization']['proposed']:.2%}")
        
        print(f"\n📁 Outputs saved to:")
        print(f"  • Models: {config.MODEL_PATH}")
        print(f"  • Plots: {config.RESULTS_PATH}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║     AMM Simulation - Use Your ETH Data                        ║
    ╚════════════════════════════════════════════════════════════════╝
    
    This script will:
    1. Verify your CSV format
    2. Run the full AMM simulation
    3. Generate comparison results
    
    Required CSV format:
        timestamp,open,high,low,close,volume
        1502901900,300.0,300.0,300.0,300.0,0.02
        1502901960,300.0,300.0,300.0,300.0,0.0
        ...
    
    Where timestamp is Unix time (seconds since Jan 1, 1970)
    """)
    
    # Get CSV path
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = input("Enter path to your CSV file: ").strip()
    
    # Remove quotes if user copied path with quotes
    csv_path = csv_path.strip('"').strip("'")
    
    if not os.path.exists(csv_path):
        print(f"\n❌ Error: File not found: {csv_path}")
        sys.exit(1)
    
    run_with_your_data(csv_path)
