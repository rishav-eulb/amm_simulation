"""
Q-Learning Reinforcement Learning Model
Implements DD-DQN (Dueling Double Deep Q-Network) for optimal liquidity provision
Following Algorithm 3 from the paper
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import random
import config


class DuelingDQN:
    """
    Dueling Double Deep Q-Network for liquidity provision optimization
    Architecture: 2 CNN layers + Dueling streams (Value + Advantage)
    """
    
    def __init__(self, state_size: int, action_size: int = 2):
        """
        Initialize DD-DQN
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
        """
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters from paper
        self.gamma = config.DISCOUNT_FACTOR
        self.epsilon = config.EPSILON_START
        self.epsilon_min = config.EPSILON_MIN
        self.epsilon_decay = config.EPSILON_DECAY
        self.learning_rate = config.Q_LEARNING_RATE
        
        # Experience replay buffer
        self.memory = deque(maxlen=10000)
        self.batch_size = config.Q_BATCH_SIZE
        
        # Build models
        self.model = self._build_model()  # Main network
        self.target_model = self._build_model()  # Target network
        self.update_target_model()
        
    def _build_model(self) -> keras.Model:
        """
        Build Dueling DQN architecture
        Following the paper's specification:
        - 2 CNN layers (100 neurons each)
        - Dueling streams: Value (50 neurons) and Advantage (50 neurons)
        - Leaky ReLU activation
        """
        # Input
        state_input = layers.Input(shape=(self.state_size,))
        
        # First dense layer (acts as CNN equivalent for 1D data)
        x = layers.Dense(
            config.DQN_LAYER1_UNITS,
            activation=layers.LeakyReLU(alpha=0.01)
        )(state_input)
        
        # Second dense layer
        x = layers.Dense(
            config.DQN_LAYER2_UNITS,
            activation=layers.LeakyReLU(alpha=0.01)
        )(x)
        
        # Dueling streams
        # Value stream
        value_stream = layers.Dense(
            config.DQN_VALUE_UNITS,
            activation=layers.LeakyReLU(alpha=0.01)
        )(x)
        value = layers.Dense(1, activation='linear')(value_stream)
        
        # Advantage stream
        advantage_stream = layers.Dense(
            config.DQN_ADVANTAGE_UNITS,
            activation=layers.LeakyReLU(alpha=0.01)
        )(x)
        advantage = layers.Dense(self.action_size, activation='linear')(advantage_stream)
        
        # Combine streams: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        # This is the key innovation of Dueling DQN
        q_values = layers.Add()([
            value,
            layers.Subtract()([
                advantage,
                layers.Lambda(lambda a: tf.reduce_mean(a, axis=1, keepdims=True))(advantage)
            ])
        ])
        
        # Build model
        model = keras.Model(inputs=state_input, outputs=q_values)
        
        # Compile with Adam optimizer
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def update_target_model(self):
        """Copy weights from main model to target model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: Whether in training mode (uses epsilon-greedy)
            
        Returns:
            Action index
        """
        if training and np.random.random() <= self.epsilon:
            # Explore: random action
            return random.randrange(self.action_size)
        
        # Exploit: best action according to Q-values
        state = state.reshape(1, -1)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self) -> float:
        """
        Train on batch from experience replay
        Implements Double Q-learning to reduce overestimation
        
        Returns:
            Training loss
        """
        if len(self.memory) < self.batch_size:
            return 0
        
        # Sample batch
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        # Double Q-learning:
        # 1. Use main network to select best action
        # 2. Use target network to evaluate that action
        
        # Get Q-values for next states from main network
        next_q_values_main = self.model.predict(next_states, verbose=0)
        best_actions = np.argmax(next_q_values_main, axis=1)
        
        # Get Q-values for next states from target network
        next_q_values_target = self.target_model.predict(next_states, verbose=0)
        
        # Calculate target Q-values
        target_q_values = rewards + (1 - dones) * self.gamma * \
                         next_q_values_target[np.arange(self.batch_size), best_actions]
        
        # Get current Q-values
        current_q_values = self.model.predict(states, verbose=0)
        
        # Update Q-values for taken actions
        for i in range(self.batch_size):
            current_q_values[i][actions[i]] = target_q_values[i]
        
        # Train model
        history = self.model.fit(
            states,
            current_q_values,
            epochs=1,
            verbose=0,
            batch_size=self.batch_size
        )
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return history.history['loss'][0]
    
    def load(self, filepath: str):
        """Load model weights"""
        self.model.load_weights(filepath)
        self.update_target_model()
    
    def save(self, filepath: str):
        """Save model weights"""
        self.model.save_weights(filepath)


class QLearningAgent:
    """
    Q-learning agent for AMM liquidity provision
    Implements the hybrid LSTM-Q-learning approach from Algorithm 3
    """
    
    def __init__(self, state_size: int = None):
        """
        Initialize Q-learning agent
        
        Args:
            state_size: Size of state space
                       (predicted valuation + equilibrium price + load)
        """
        if state_size is None:
            # State: [v'_p predictions (window), v', E_p[load]]
            state_size = config.Q_WINDOW_SIZE + 2
        
        self.state_size = state_size
        
        # Actions: 0 = Do nothing, 1 = Insert Gaussian parameter ε
        self.action_size = 2
        
        # Build DD-DQN
        self.dqn = DuelingDQN(state_size, self.action_size)
        
        # Gaussian parameter distribution (for action 1)
        self.gaussian_mu = 0
        self.gaussian_sigma = 0.1
        
        # Threshold for loss tolerance
        self.beta_c = config.BETA_C
        
    def prepare_state(self, predicted_valuations: np.ndarray,
                     equilibrium_valuation: float,
                     expected_load: float) -> np.ndarray:
        """
        Prepare state vector for Q-learning
        
        Args:
            predicted_valuations: Recent v'_p predictions (window_size)
            equilibrium_valuation: Current equilibrium v'
            expected_load: Computed E_p[load(v')]
            
        Returns:
            State vector
        """
        # Ensure predicted_valuations has correct length
        if len(predicted_valuations) < config.Q_WINDOW_SIZE:
            # Pad with zeros if needed
            padding = np.zeros(config.Q_WINDOW_SIZE - len(predicted_valuations))
            predicted_valuations = np.concatenate([padding, predicted_valuations])
        else:
            predicted_valuations = predicted_valuations[-config.Q_WINDOW_SIZE:]
        
        # Combine into state vector
        state = np.concatenate([
            predicted_valuations,
            [equilibrium_valuation],
            [expected_load]
        ])
        
        return state
    
    def calculate_reward(self, loss: float) -> int:
        """
        Calculate reward based on loss
        Following the paper's reward function
        
        Args:
            loss: |v'_t - v'_p,t| + E_p[load(v')]
            
        Returns:
            Reward: -1, 0, or +1
        """
        if loss > self.beta_c:
            return -1
        elif loss == self.beta_c:
            return 0
        else:
            return +1
    
    def get_action(self, state: np.ndarray, training: bool = True) -> tuple:
        """
        Get action from agent
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            (action_type, gaussian_param): Action type and Gaussian parameter
        """
        action = self.dqn.act(state, training)
        
        if action == 1:
            # Generate Gaussian parameter ε
            epsilon = np.random.normal(self.gaussian_mu, self.gaussian_sigma)
            # Clip to [-1, 1]
            epsilon = np.clip(epsilon, -1, 1)
        else:
            epsilon = 0
        
        return action, epsilon
    
    def train_step(self, state: np.ndarray, action: int, reward: int,
                   next_state: np.ndarray, done: bool) -> float:
        """
        Perform one training step
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            
        Returns:
            Training loss
        """
        # Remember experience
        self.dqn.remember(state, action, reward, next_state, done)
        
        # Train on replay buffer
        loss = self.dqn.replay()
        
        return loss
    
    def update_target_network(self):
        """Update target network weights"""
        self.dqn.update_target_model()
    
    def save_model(self, filepath: str):
        """Save model"""
        self.dqn.save(filepath)
    
    def load_model(self, filepath: str):
        """Load model"""
        self.dqn.load(filepath)


def calculate_loss(predicted_valuation: float,
                   equilibrium_valuation: float,
                   expected_load: float) -> float:
    """
    Calculate loss function from Equation 3
    ℓ := |v'_t - v'_p,t| + E_p[load(v')]
    
    Args:
        predicted_valuation: Predicted v'_p
        equilibrium_valuation: Equilibrium v'
        expected_load: Expected load
        
    Returns:
        Loss value
    """
    prediction_slippage = abs(equilibrium_valuation - predicted_valuation)
    total_loss = prediction_slippage + expected_load
    
    return total_loss


def create_qlearning_agent(state_size: int = None) -> QLearningAgent:
    """
    Factory function to create Q-learning agent
    
    Args:
        state_size: Size of state space
        
    Returns:
        QLearningAgent instance
    """
    return QLearningAgent(state_size)
