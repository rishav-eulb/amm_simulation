"""
Configuration for Predictive AMM Simulation
Based on: Predictive crypto-asset automated market maker architecture 
for decentralized finance using deep reinforcement learning
"""

# ============================================================================
# MARKET PARAMETERS
# ============================================================================

# Liquidity constant for constant product market maker (x * y = c)
LIQUIDITY_CONSTANT = 100_000_000  # 100 million (realistic for ETH pool)

# Fee tier (as decimal)
FEE_TIER = 0.003  # 0.3% (Uniswap V3 standard)

# ============================================================================
# DATA PARAMETERS
# ============================================================================

# Time interval for data
DATA_INTERVAL = '1min'  # 1-minute candles

# Train/Test/Validation split (total: ~72k hours = 4.3M minutes)
TRAIN_HOURS = 14000   # 70% of data (3M minutes)
TEST_HOURS = 4000    # 21% of data (900k minutes)
VALIDATION_HOURS = 2000  # 9% of data (420k minutes)

# ============================================================================
# LSTM PARAMETERS (from paper)
# ============================================================================

# LSTM architecture
LSTM_UNITS = 100
LSTM_WINDOW_SIZE = 50  # Number of historical intervals to look back

# Training parameters
LSTM_EPOCHS = 50  # Reduced from 50 (good enough, saves 5x time)
LSTM_BATCH_SIZE = 32
LSTM_LEARNING_RATE = 0.001

# ============================================================================
# Q-LEARNING PARAMETERS (from paper)
# ============================================================================

# Q-learning hyperparameters
Q_LEARNING_RATE = 0.001  # Conservative for stability (0.001 risks explosion)
DISCOUNT_FACTOR = 0.98  # gamma
EPSILON_START = 1.0  # Exploration rate
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9995

# DD-DQN architecture
DQN_LAYER1_UNITS = 100
DQN_LAYER2_UNITS = 100
DQN_VALUE_UNITS = 50
DQN_ADVANTAGE_UNITS = 50

# Training parameters
Q_EPOCHS = 50
Q_BATCH_SIZE = 16
Q_WINDOW_SIZE = 10  # For Q-learning input

# ============================================================================
# REWARD FUNCTION PARAMETERS
# ============================================================================

# Threshold for prediction slippage and load tolerance
BETA_C = 500  # β_c threshold for acceptable loss
BETA_V = 0.00001  # β_v threshold for valuation-based events (event-step threshold in valuation space)

# Monte Carlo and valuation parameters
LOAD_MC_SAMPLES = 64  # Monte Carlo samples for E_p[load]
PRED_NOISE_SCALE = 0.75  # scales sigma from LSTM val error to build p(v')

# Valuation bounds (for numerical stability)
MIN_V = 1e-6
MAX_V = 1 - 1e-6

# ============================================================================
# LIQUIDITY PROVISION PARAMETERS
# ============================================================================

# Gaussian incentive fee distribution
INCENTIVE_STD_DEV = 0.05  # σ_φ for Gaussian distribution

# Number of intervals to predict ahead
PREDICTION_INTERVALS = [1, 5, 10]  # n intervals for predictive liquidity

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

# Initial token reserves (will be calculated based on c and initial price)
INITIAL_PRICE = None  # Will be set from first data point

# Trade volume parameters (for synthetic trade generation)
MIN_TRADE_SIZE = 0.01  # Minimum trade size as % of pool
MAX_TRADE_SIZE = 5.0   # Maximum trade size as % of pool
TRADE_FREQUENCY = 0.3  # Probability of trade per interval

# ============================================================================
# EVALUATION METRICS
# ============================================================================

# Price impact threshold for liquidity depth evaluation
PRICE_IMPACT_THRESHOLD = 0.01  # 1% acceptable threshold

# ============================================================================
# PATHS
# ============================================================================

import os
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_BASE_DIR, 'data')
MODEL_PATH = os.path.join(_BASE_DIR, 'models')
RESULTS_PATH = os.path.join(_BASE_DIR, 'results')

# ============================================================================
# ADDITIONAL FEATURES
# ============================================================================

# Enable alternative data sources (e.g., Twitter sentiment)
USE_ALTERNATIVE_DATA = False

# Random seed for reproducibility
RANDOM_SEED = 42
