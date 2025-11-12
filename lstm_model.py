"""
LSTM Price Prediction Model
Implements the LSTM component of the hybrid LSTM-Q-learning architecture
Following Algorithm 2 from the paper
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, List
import config


class LSTMPricePredictor:
    """
    LSTM model for predicting future market valuations
    Architecture: 1 LSTM layer with 100 units, window size of 50 intervals
    """
    
    def __init__(self, window_size: int = None, lstm_units: int = None):
        """
        Initialize LSTM predictor
        
        Args:
            window_size: Number of historical intervals to use
            lstm_units: Number of LSTM units
        """
        self.window_size = window_size or config.LSTM_WINDOW_SIZE
        self.lstm_units = lstm_units or config.LSTM_UNITS
        
        self.model = None
        self.history = None
        self.scaler = None  # For price normalization
        
    def build_model(self, input_features: int = 3):
        """
        Build LSTM model architecture
        
        Args:
            input_features: Number of input features
                          (price, alternative data signals, Gaussian parameter)
        """
        model = keras.Sequential([
            # LSTM layer with 100 units
            layers.LSTM(
                units=self.lstm_units,
                activation='tanh',
                return_sequences=False,
                input_shape=(self.window_size, input_features)
            ),
            
            # Output layer for price prediction
            layers.Dense(1, activation='linear')
        ])
        
        # Compile with Adam optimizer (as specified in paper)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.LSTM_LEARNING_RATE),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        
        return model
    
    def prepare_sequences(self, prices: np.ndarray, 
                         alternative_signals: np.ndarray = None,
                         gaussian_params: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sliding window sequences for LSTM training
        
        Args:
            prices: Array of historical prices (v_obs)
            alternative_signals: Alternative data signals (τ)
            gaussian_params: Gaussian input parameters from Q-learning (ε)
            
        Returns:
            (X, y): Input sequences and target values
        """
        n_samples = len(prices) - self.window_size
        
        # Initialize features array
        if alternative_signals is None:
            alternative_signals = np.zeros_like(prices)
        if gaussian_params is None:
            gaussian_params = np.zeros_like(prices)
        
        # Create sliding windows
        X = []
        y = []
        
        for i in range(n_samples):
            # Input window: [price, alt_signals, gaussian_params]
            window_prices = prices[i:i+self.window_size]
            window_alt = alternative_signals[i:i+self.window_size]
            window_gauss = gaussian_params[i:i+self.window_size]
            
            # Stack features
            window_features = np.column_stack([
                window_prices,
                window_alt,
                window_gauss
            ])
            
            X.append(window_features)
            
            # Target: next price
            y.append(prices[i+self.window_size])
        
        X = np.array(X)
        y = np.array(y)
        
        return X, y
    
    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data using min-max scaling
        
        Args:
            data: Input data
            
        Returns:
            Normalized data
        """
        from sklearn.preprocessing import MinMaxScaler
        
        if self.scaler is None:
            self.scaler = MinMaxScaler()
            normalized = self.scaler.fit_transform(data.reshape(-1, 1))
        else:
            normalized = self.scaler.transform(data.reshape(-1, 1))
        
        return normalized.flatten()
    
    def denormalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        Denormalize data back to original scale
        
        Args:
            data: Normalized data
            
        Returns:
            Original scale data
        """
        if self.scaler is None:
            raise ValueError("Must fit scaler before denormalizing")
        
        return self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()
    
    def train(self, prices: np.ndarray,
              alternative_signals: np.ndarray = None,
              gaussian_params: np.ndarray = None,
              epochs: int = None,
              batch_size: int = None,
              validation_split: float = 0.2) -> keras.callbacks.History:
        """
        Train LSTM model
        Following Algorithm 2 from the paper
        
        Args:
            prices: Historical prices
            alternative_signals: Alternative data signals
            gaussian_params: Gaussian parameters from Q-learning
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation data split
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        epochs = epochs or config.LSTM_EPOCHS
        batch_size = batch_size or config.LSTM_BATCH_SIZE
        
        # Normalize prices
        prices_normalized = self.normalize_data(prices)
        
        # Prepare sequences
        X, y = self.prepare_sequences(
            prices_normalized,
            alternative_signals,
            gaussian_params
        )
        
        # Train model
        self.history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        return self.history
    
    def predict(self, recent_prices: np.ndarray,
                recent_alt_signals: np.ndarray = None,
                recent_gauss_params: np.ndarray = None) -> float:
        """
        Predict next price (v'_p)
        
        Args:
            recent_prices: Recent price history (window_size length)
            recent_alt_signals: Recent alternative signals
            recent_gauss_params: Recent Gaussian parameters
            
        Returns:
            Predicted next price
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        if len(recent_prices) < self.window_size:
            raise ValueError(f"Need at least {self.window_size} price points")
        
        # Take last window_size prices
        recent_prices = recent_prices[-self.window_size:]
        
        # Normalize
        prices_normalized = self.normalize_data(recent_prices)
        
        # Prepare input
        if recent_alt_signals is None:
            recent_alt_signals = np.zeros_like(recent_prices)
        else:
            recent_alt_signals = recent_alt_signals[-self.window_size:]
        
        if recent_gauss_params is None:
            recent_gauss_params = np.zeros_like(recent_prices)
        else:
            recent_gauss_params = recent_gauss_params[-self.window_size:]
        
        # Stack features
        input_features = np.column_stack([
            prices_normalized,
            recent_alt_signals,
            recent_gauss_params
        ])
        
        # Reshape for model input: (1, window_size, features)
        input_data = input_features.reshape(1, self.window_size, -1)
        
        # Predict
        prediction_normalized = self.model.predict(input_data, verbose=0)[0][0]
        
        # Denormalize
        prediction = self.denormalize_data(np.array([prediction_normalized]))[0]
        
        return prediction
    
    def predict_multi_step(self, recent_prices: np.ndarray,
                           n_steps: int = 10) -> np.ndarray:
        """
        Predict multiple steps ahead
        
        Args:
            recent_prices: Recent price history
            n_steps: Number of steps to predict ahead
            
        Returns:
            Array of predicted prices
        """
        predictions = []
        current_window = recent_prices[-self.window_size:].copy()
        
        for _ in range(n_steps):
            # Predict next step
            next_price = self.predict(current_window)
            predictions.append(next_price)
            
            # Update window (slide forward)
            current_window = np.append(current_window[1:], next_price)
        
        return np.array(predictions)
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if self.model is not None:
            self.model.save(filepath)
    
    def load_model(self, filepath: str):
        """Load trained model"""
        self.model = keras.models.load_model(filepath)
    
    def get_prediction_error(self, actual_prices: np.ndarray,
                            test_start_idx: int) -> dict:
        """
        Calculate prediction errors on test set
        
        Args:
            actual_prices: Actual price series
            test_start_idx: Index where test set starts
            
        Returns:
            Dictionary with error metrics
        """
        test_prices = actual_prices[test_start_idx:]
        predictions = []
        actuals = []
        
        for i in range(len(test_prices) - self.window_size):
            # Get window
            window = test_prices[i:i+self.window_size]
            
            # Predict
            pred = self.predict(window)
            predictions.append(pred)
            
            # Actual
            actual = test_prices[i+self.window_size]
            actuals.append(actual)
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals)**2))
        mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'predictions': predictions,
            'actuals': actuals
        }


def create_lstm_predictor(window_size: int = None,
                          lstm_units: int = None) -> LSTMPricePredictor:
    """
    Factory function to create LSTM predictor
    
    Args:
        window_size: Size of sliding window
        lstm_units: Number of LSTM units
        
    Returns:
        LSTMPricePredictor instance
    """
    return LSTMPricePredictor(window_size, lstm_units)
