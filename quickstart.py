#!/usr/bin/env python3
"""
Quick Start Script for AMM Simulation
Run this to immediately start the simulation with default settings
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation import run_eth_simulation
import config

def main():
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║       Predictive AMM Simulation for ETH - Quick Start         ║
    ╚════════════════════════════════════════════════════════════════╝
    
    This simulation will:
    1. Generate synthetic ETH-like price data (1-minute intervals)
    2. Train LSTM model for price prediction
    3. Train Q-learning agent for optimal liquidity provision
    4. Compare proposed AMM vs baseline Uniswap V3
    5. Generate evaluation metrics and plots
    
    Configuration:
    - Liquidity Constant (c): {:,}
    - Training Data: {} hours
    - Test Data: {} hours
    - Initial ETH Price: ~$2000
    
    Press Enter to start or Ctrl+C to cancel...
    """.format(
        config.LIQUIDITY_CONSTANT,
        config.TRAIN_HOURS,
        config.TEST_HOURS
    ))
    
    try:
        input()
    except KeyboardInterrupt:
        print("\n\nSimulation cancelled.")
        return
    
    print("\n" + "="*70)
    print("Starting simulation...")
    print("="*70 + "\n")
    
    # Run simulation
    try:
        results = run_eth_simulation(use_synthetic=True)
        
        print("\n" + "="*70)
        print("SUCCESS! Simulation completed.")
        print("="*70)
        print(f"\nOutputs saved to:")
        print(f"  - Models: {config.MODEL_PATH}")
        print(f"  - Results: {config.RESULTS_PATH}")
        print(f"\nKey Results:")
        print(f"  - Divergence Loss Improvement: {results['divergence_loss']['improvement']:.2f}%")
        print(f"  - Slippage Improvement: {results['slippage']['improvement']:.2f}%")
        print(f"  - Liquidity Utilization (Proposed): {results['liquidity_utilization']['proposed']:.2%}")
        
    except Exception as e:
        print(f"\n❌ Error during simulation: {e}")
        print("\nDebug information:")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
