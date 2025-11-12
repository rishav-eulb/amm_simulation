"""
Main Simulation Runner
Integrates LSTM, Q-learning, and AMM components for complete simulation
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import os

import config
from amm_core import create_amm, ConfigurableVirtualAMM, CPMM
from lstm_model import create_lstm_predictor
from qlearning_model import create_qlearning_agent, calculate_loss
from data_utils import DataLoader, TradeGenerator


class AMMSimulation:
    """Complete AMM simulation with predictive capabilities"""
    
    def __init__(self, initial_price: float, liquidity_constant: float = None):
        """
        Initialize simulation
        
        Args:
            initial_price: Initial ETH price
            liquidity_constant: Liquidity constant c (default from config)
        """
        self.initial_price = initial_price
        self.liquidity_constant = liquidity_constant or config.LIQUIDITY_CONSTANT
        
        # Create AMM instances
        self.proposed_amm = create_amm(
            self.liquidity_constant,
            initial_price,
            use_predictive=True
        )
        
        self.baseline_amm = create_amm(
            self.liquidity_constant,
            initial_price,
            use_predictive=False
        )
        
        # Create ML models
        self.lstm_model = create_lstm_predictor()
        self.qlearning_agent = create_qlearning_agent()
        
        # Initialize tracking
        self.results = {
            'proposed': {
                'divergence_losses': [],
                'slippage_losses': [],
                'liquidity_utilizations': [],
                'prices': [],
                'reserves_x': [],
                'reserves_y': []
            },
            'baseline': {
                'divergence_losses': [],
                'slippage_losses': [],
                'liquidity_utilizations': [],
                'prices': [],
                'reserves_x': [],
                'reserves_y': []
            }
        }
        
    def train_lstm(self, train_prices: np.ndarray,
                   alternative_signals: np.ndarray = None) -> dict:
        """
        Train LSTM price predictor
        
        Args:
            train_prices: Training price data
            alternative_signals: Alternative data signals
            
        Returns:
            Training history
        """
        print("Training LSTM model...")
        
        history = self.lstm_model.train(
            train_prices,
            alternative_signals=alternative_signals,
            epochs=config.LSTM_EPOCHS,
            batch_size=config.LSTM_BATCH_SIZE
        )
        
        print("LSTM training complete")
        return history.history
    
    def train_qlearning(self, train_prices: np.ndarray,
                       n_episodes: int = 100) -> list:
        """
        Train Q-learning agent
        
        Args:
            train_prices: Training price data
            n_episodes: Number of training episodes
            
        Returns:
            List of episode losses
        """
        print("Training Q-learning agent...")
        
        episode_losses = []
        
        for episode in range(n_episodes):
            episode_loss = 0
            n_steps = 0
            
            # Sample random starting point
            start_idx = np.random.randint(
                config.LSTM_WINDOW_SIZE,
                len(train_prices) - config.Q_WINDOW_SIZE - 1
            )
            
            for t in range(start_idx, min(start_idx + 1000, len(train_prices) - 1)):
                # Get LSTM predictions for window
                window_prices = train_prices[max(0, t - config.LSTM_WINDOW_SIZE):t]
                
                if len(window_prices) >= config.LSTM_WINDOW_SIZE:
                    predicted_price = self.lstm_model.predict(window_prices)
                    
                    # Calculate equilibrium valuation
                    current_price = train_prices[t]
                    equilibrium_v = 1 / (1 + current_price**2)
                    predicted_v = 1 / (1 + predicted_price**2)
                    
                    # Calculate expected load (simplified)
                    expected_load = abs(predicted_v - equilibrium_v) * 0.1
                    
                    # Prepare state
                    recent_predictions = [predicted_v] * config.Q_WINDOW_SIZE
                    state = self.qlearning_agent.prepare_state(
                        np.array(recent_predictions),
                        equilibrium_v,
                        expected_load
                    )
                    
                    # Get action
                    action, epsilon = self.qlearning_agent.get_action(state, training=True)
                    
                    # Calculate loss and reward
                    loss = calculate_loss(predicted_v, equilibrium_v, expected_load)
                    reward = self.qlearning_agent.calculate_reward(loss)
                    
                    # Next state
                    next_price = train_prices[t + 1]
                    next_equilibrium_v = 1 / (1 + next_price**2)
                    next_state = self.qlearning_agent.prepare_state(
                        np.array(recent_predictions),
                        next_equilibrium_v,
                        expected_load
                    )
                    
                    # Train
                    step_loss = self.qlearning_agent.train_step(
                        state, action, reward, next_state, False
                    )
                    
                    episode_loss += step_loss
                    n_steps += 1
            
            # Update target network periodically
            if episode % 10 == 0:
                self.qlearning_agent.update_target_network()
            
            avg_loss = episode_loss / max(n_steps, 1)
            episode_losses.append(avg_loss)
            
            if episode % 10 == 0:
                print(f"Episode {episode}/{n_episodes}, Avg Loss: {avg_loss:.4f}, "
                      f"Epsilon: {self.qlearning_agent.dqn.epsilon:.4f}")
        
        print("Q-learning training complete")
        return episode_losses
    
    def simulate_episode(self, prices: np.ndarray,
                        trades: list,
                        use_proposed: bool = True) -> Dict:
        """
        Simulate one episode of trading
        
        Args:
            prices: Price series
            trades: List of (interval, size, is_buy) tuples
            use_proposed: Whether to use proposed AMM (vs baseline)
            
        Returns:
            Episode results
        """
        amm = self.proposed_amm if use_proposed else self.baseline_amm
        results_key = 'proposed' if use_proposed else 'baseline'
        
        # Reset AMM
        amm.x = amm.initial_x
        amm.y = amm.initial_y
        
        # Track metrics
        divergence_losses = []
        slippage_losses = []
        prices_tracked = []
        
        # Group trades by interval
        trades_by_interval = {}
        for interval, size, is_buy in trades:
            if interval not in trades_by_interval:
                trades_by_interval[interval] = []
            trades_by_interval[interval].append((size, is_buy))
        
        # Simulate each interval
        for t in range(len(prices)):
            current_price = prices[t]
            
            # If using proposed AMM, apply pseudo-arbitrage
            if use_proposed and t > 0:
                amm.pseudo_arbitrage(current_price)
            
            # Execute trades for this interval
            if t in trades_by_interval:
                for size, is_buy in trades_by_interval[t]:
                    if is_buy:
                        # Buy X with Y
                        received = amm.swap_y_for_x(
                            size * current_price,
                            fee=config.FEE_TIER
                        )
                    else:
                        # Sell X for Y
                        received = amm.swap_x_for_y(
                            size,
                            fee=config.FEE_TIER
                        )
                    
                    # Calculate slippage for this trade
                    slippage = amm.calculate_slippage(size, is_buy, config.FEE_TIER)
                    slippage_losses.append(slippage)
            
            # Calculate divergence loss
            initial_value = amm.initial_x * current_price + amm.initial_y
            div_loss = amm.calculate_divergence_loss(initial_value, current_price)
            divergence_losses.append(div_loss)
            
            # Track price
            prices_tracked.append(amm.get_price())
            
            # Track reserves
            self.results[results_key]['reserves_x'].append(amm.x)
            self.results[results_key]['reserves_y'].append(amm.y)
        
        return {
            'divergence_losses': divergence_losses,
            'slippage_losses': slippage_losses,
            'prices': prices_tracked,
            'final_reserves': (amm.x, amm.y)
        }
    
    def run_simulation(self, train_prices: np.ndarray,
                      test_prices: np.ndarray) -> Dict:
        """
        Run complete simulation: train models, then evaluate
        
        Args:
            train_prices: Training data
            test_prices: Test data
            
        Returns:
            Complete simulation results
        """
        print("="*60)
        print("Starting AMM Simulation")
        print("="*60)
        
        # Step 1: Train LSTM
        lstm_history = self.train_lstm(train_prices)
        
        # Step 2: Train Q-learning
        qlearning_losses = self.train_qlearning(train_prices, n_episodes=50)
        
        # Step 3: Generate trades
        print("\nGenerating synthetic trades...")
        trade_gen = TradeGenerator(np.sqrt(self.liquidity_constant))
        trades = trade_gen.generate_market_impact_trades(test_prices)
        print(f"Generated {len(trades)} trades")
        
        # Step 4: Simulate proposed AMM
        print("\nSimulating proposed AMM...")
        proposed_results = self.simulate_episode(test_prices, trades, use_proposed=True)
        
        # Step 5: Simulate baseline AMM
        print("Simulating baseline AMM...")
        baseline_results = self.simulate_episode(test_prices, trades, use_proposed=False)
        
        # Step 6: Calculate metrics
        results = self.calculate_metrics(proposed_results, baseline_results, test_prices)
        
        print("\n" + "="*60)
        print("Simulation Complete")
        print("="*60)
        
        return results
    
    def calculate_metrics(self, proposed_results: Dict,
                         baseline_results: Dict,
                         test_prices: np.ndarray) -> Dict:
        """
        Calculate evaluation metrics
        
        Args:
            proposed_results: Results from proposed AMM
            baseline_results: Results from baseline AMM
            test_prices: Test price series
            
        Returns:
            Metrics dictionary
        """
        # Calculate average divergence loss
        proposed_div = np.mean(proposed_results['divergence_losses'])
        baseline_div = np.mean(baseline_results['divergence_losses'])
        
        # Calculate average slippage
        proposed_slip = np.mean(proposed_results['slippage_losses']) if proposed_results['slippage_losses'] else 0
        baseline_slip = np.mean(baseline_results['slippage_losses']) if baseline_results['slippage_losses'] else 0
        
        # Calculate liquidity utilization
        total_volume = len(proposed_results['slippage_losses']) * np.sqrt(self.liquidity_constant) * 0.01
        proposed_util = total_volume / np.sqrt(self.liquidity_constant)
        baseline_util = total_volume / np.sqrt(self.liquidity_constant)
        
        # Calculate liquidity depth
        proposed_depth = self.proposed_amm.calculate_liquidity_depth()
        baseline_depth = self.baseline_amm.calculate_liquidity_depth()
        
        metrics = {
            'divergence_loss': {
                'proposed': proposed_div,
                'baseline': baseline_div,
                'improvement': (baseline_div - proposed_div) / baseline_div * 100
            },
            'slippage': {
                'proposed': proposed_slip,
                'baseline': baseline_slip,
                'improvement': (baseline_slip - proposed_slip) / baseline_slip * 100 if baseline_slip > 0 else 0
            },
            'liquidity_utilization': {
                'proposed': proposed_util,
                'baseline': baseline_util
            },
            'liquidity_depth': {
                'proposed': proposed_depth,
                'baseline': baseline_depth
            }
        }
        
        self.print_metrics(metrics)
        
        return metrics
    
    def print_metrics(self, metrics: Dict):
        """Print metrics in formatted way"""
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)
        
        print("\n1. DIVERGENCE LOSS:")
        print(f"   Proposed AMM: {metrics['divergence_loss']['proposed']:.4f}")
        print(f"   Baseline AMM: {metrics['divergence_loss']['baseline']:.4f}")
        print(f"   Improvement:  {metrics['divergence_loss']['improvement']:.2f}%")
        
        print("\n2. SLIPPAGE:")
        print(f"   Proposed AMM: {metrics['slippage']['proposed']:.4f}")
        print(f"   Baseline AMM: {metrics['slippage']['baseline']:.4f}")
        print(f"   Improvement:  {metrics['slippage']['improvement']:.2f}%")
        
        print("\n3. LIQUIDITY UTILIZATION:")
        print(f"   Proposed AMM: {metrics['liquidity_utilization']['proposed']:.2%}")
        print(f"   Baseline AMM: {metrics['liquidity_utilization']['baseline']:.2%}")
        
        print("\n4. LIQUIDITY DEPTH:")
        print(f"   Proposed AMM: {metrics['liquidity_depth']['proposed']:.2f}")
        print(f"   Baseline AMM: {metrics['liquidity_depth']['baseline']:.2f}")
        
        print("="*60)
    
    def plot_results(self, save_path: str = None):
        """
        Plot simulation results
        
        Args:
            save_path: Path to save plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Divergence loss comparison
        ax = axes[0, 0]
        proposed_div = self.results['proposed']['divergence_losses']
        baseline_div = self.results['baseline']['divergence_losses']
        
        if proposed_div and baseline_div:
            ax.plot(proposed_div, label='Proposed AMM', alpha=0.7)
            ax.plot(baseline_div, label='Baseline AMM', alpha=0.7)
            ax.set_title('Divergence Loss Over Time')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Divergence Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 2: Price comparison
        ax = axes[0, 1]
        proposed_prices = self.results['proposed']['prices']
        baseline_prices = self.results['baseline']['prices']
        
        if proposed_prices and baseline_prices:
            ax.plot(proposed_prices, label='Proposed AMM', alpha=0.7)
            ax.plot(baseline_prices, label='Baseline AMM', alpha=0.7)
            ax.set_title('AMM Prices Over Time')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Reserves X
        ax = axes[1, 0]
        proposed_x = self.results['proposed']['reserves_x']
        baseline_x = self.results['baseline']['reserves_x']
        
        if proposed_x and baseline_x:
            ax.plot(proposed_x, label='Proposed AMM', alpha=0.7)
            ax.plot(baseline_x, label='Baseline AMM', alpha=0.7)
            ax.set_title('Token X Reserves Over Time')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Reserves X')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Reserves Y
        ax = axes[1, 1]
        proposed_y = self.results['proposed']['reserves_y']
        baseline_y = self.results['baseline']['reserves_y']
        
        if proposed_y and baseline_y:
            ax.plot(proposed_y, label='Proposed AMM', alpha=0.7)
            ax.plot(baseline_y, label='Baseline AMM', alpha=0.7)
            ax.set_title('Token Y Reserves Over Time')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Reserves Y')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to: {save_path}")
        else:
            plt.savefig('/home/claude/amm_simulation/results/simulation_results.png',
                       dpi=300, bbox_inches='tight')
            print("\nPlot saved to: /home/claude/amm_simulation/results/simulation_results.png")
        
        plt.close()
    
    def save_models(self, directory: str = None):
        """
        Save trained models
        
        Args:
            directory: Directory to save models
        """
        if directory is None:
            directory = config.MODEL_PATH
        
        os.makedirs(directory, exist_ok=True)
        
        # Save LSTM
        lstm_path = os.path.join(directory, 'lstm_model.keras')
        self.lstm_model.save_model(lstm_path)
        print(f"LSTM model saved to: {lstm_path}")
        
        # Save Q-learning
        q_path = os.path.join(directory, 'qlearning_model.weights.h5')
        self.qlearning_agent.save_model(q_path)
        print(f"Q-learning model saved to: {q_path}")


def run_eth_simulation(eth_data_path: str = None,
                       use_synthetic: bool = True) -> Dict:
    """
    Main entry point to run ETH AMM simulation
    
    Args:
        eth_data_path: Path to ETH price CSV
        use_synthetic: Whether to use synthetic data
        
    Returns:
        Simulation results
    """
    # Create output directories
    os.makedirs(config.DATA_PATH, exist_ok=True)
    os.makedirs(config.MODEL_PATH, exist_ok=True)
    os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    # Load data
    if use_synthetic or eth_data_path is None:
        print("Generating synthetic ETH-like data...")
        from data_utils import create_synthetic_eth_data
        train_df, test_df, val_df = create_synthetic_eth_data()
    else:
        print(f"Loading ETH data from: {eth_data_path}")
        from data_utils import load_eth_data
        train_df, test_df, val_df = load_eth_data(eth_data_path)
    
    train_prices = train_df['close'].values
    test_prices = test_df['close'].values
    
    initial_price = train_prices[0]
    
    print(f"\nData loaded:")
    print(f"  Train samples: {len(train_prices)} ({len(train_prices)/60:.1f} hours)")
    print(f"  Test samples:  {len(test_prices)} ({len(test_prices)/60:.1f} hours)")
    print(f"  Initial price: ${initial_price:.2f}")
    print(f"  Liquidity constant: {config.LIQUIDITY_CONSTANT:,.0f}")
    
    # Create and run simulation
    simulation = AMMSimulation(initial_price, config.LIQUIDITY_CONSTANT)
    results = simulation.run_simulation(train_prices, test_prices)
    
    # Save models
    simulation.save_models()
    
    # Plot results
    simulation.plot_results()
    
    return results


if __name__ == "__main__":
    # Run simulation with synthetic data
    results = run_eth_simulation(use_synthetic=False)
