"""
Core Automated Market Maker (AMM) Implementation
Implements CPMM (Constant Product Market Maker) and configurable virtual AMM
"""

import numpy as np
from typing import Tuple, Optional
import config


class CPMM:
    """
    Constant Product Market Maker (x * y = c)
    Base implementation following Uniswap V2/V3 principles
    """
    
    def __init__(self, liquidity_constant: float, initial_price: float):
        """
        Initialize CPMM with liquidity constant and initial price.
        
        Args:
            liquidity_constant: The constant c in x*y=c formula
            initial_price: Initial price of token X in terms of token Y
        """
        self.c = liquidity_constant
        
        # Calculate initial reserves: x * y = c, and price = y/x
        # So: x * (price * x) = c => x^2 = c/price
        self.x = np.sqrt(self.c / initial_price)
        self.y = self.c / self.x
        
        self.initial_x = self.x
        self.initial_y = self.y
        
    def get_price(self) -> float:
        """Get current price of token X in terms of token Y"""
        return self.y / self.x
    
    def get_reserves(self) -> Tuple[float, float]:
        """Get current reserves"""
        return self.x, self.y
    
    def swap_x_for_y(self, delta_x: float, fee: float = 0.0) -> float:
        """
        Swap delta_x of token X for token Y
        
        Args:
            delta_x: Amount of token X to swap
            fee: Trading fee (e.g., 0.003 for 0.3%)
            
        Returns:
            Amount of token Y received
        """
        if delta_x <= 0:
            return 0
        
        # Apply fee to input
        delta_x_after_fee = delta_x * (1 - fee)
        
        # Calculate output: (x + δx) * (y - δy) = c
        # δy = y - c/(x + δx)
        new_x = self.x + delta_x_after_fee
        new_y = self.c / new_x
        delta_y = self.y - new_y
        
        # Update reserves
        self.x = new_x
        self.y = new_y
        
        return delta_y
    
    def swap_y_for_x(self, delta_y: float, fee: float = 0.0) -> float:
        """
        Swap delta_y of token Y for token X
        
        Args:
            delta_y: Amount of token Y to swap
            fee: Trading fee
            
        Returns:
            Amount of token X received
        """
        if delta_y <= 0:
            return 0
        
        delta_y_after_fee = delta_y * (1 - fee)
        
        new_y = self.y + delta_y_after_fee
        new_x = self.c / new_y
        delta_x = self.x - new_x
        
        self.x = new_x
        self.y = new_y
        
        return delta_x
    
    def get_equilibrium_valuation(self, market_price: float) -> float:
        """
        Calculate equilibrium valuation v that minimizes v·x
        Following Equation 1 from the paper
        
        Args:
            market_price: Observed market price from oracle
            
        Returns:
            Equilibrium valuation v
        """
        # For CPMM with f(x) = c/x:
        # Equilibrium when df/dx = -v/(1-v)
        # df/dx = -c/x^2, so: -c/x^2 = -v/(1-v)
        # This gives: v = c/(x^2 + c)
        
        current_price = self.get_price()
        # Using the optimization: v := min_x (v_obs · x)
        # For CPMM: φ(v) = sqrt((1-v)/v)
        # So: v = 1/(1 + current_price^2)
        
        v = 1 / (1 + current_price**2)
        return v
    
    def calculate_divergence_loss(self, initial_value: float, 
                                  current_price: float) -> float:
        """
        Calculate divergence (impermanent) loss
        Following the paper's formula
        
        Args:
            initial_value: Initial portfolio value
            current_price: Current market price
            
        Returns:
            Divergence loss amount
        """
        # Current value in the pool
        current_value = self.x * current_price + self.y
        
        # Value if held outside pool
        hold_value = self.initial_x * current_price + self.initial_y
        
        # Divergence loss
        div_loss = hold_value - current_value
        
        return max(0, div_loss)  # Loss is positive
    
    def calculate_slippage(self, trade_size: float, is_buy: bool,
                          fee: float = 0.0) -> float:
        """
        Calculate slippage for a given trade size
        
        Args:
            trade_size: Size of trade in token X
            is_buy: True if buying X, False if selling X
            fee: Trading fee
            
        Returns:
            Slippage amount
        """
        initial_price = self.get_price()
        
        # Make a copy to simulate trade without affecting state
        temp_x, temp_y = self.x, self.y
        
        if is_buy:
            # Calculate how much Y needed to get trade_size of X
            new_x = self.x - trade_size
            new_y = self.c / new_x
            delta_y = new_y - self.y
            expected_cost = trade_size * initial_price
            actual_cost = delta_y
            slippage = actual_cost - expected_cost
        else:
            # Calculate how much Y received for selling trade_size of X
            new_x = self.x + trade_size
            new_y = self.c / new_x
            delta_y = self.y - new_y
            expected_received = trade_size * initial_price
            actual_received = delta_y
            slippage = expected_received - actual_received
        
        # Restore state
        self.x, self.y = temp_x, temp_y
        
        return abs(slippage)
    
    def get_capitalization(self, valuation: float) -> float:
        """
        Calculate total value of AMM holdings at given valuation
        Following the paper's cap(x,v) formula
        
        Args:
            valuation: Market valuation v
            
        Returns:
            Total capitalization
        """
        return valuation * self.x + (1 - valuation) * self.y


class ConfigurableVirtualAMM(CPMM):
    """
    Configurable Virtual AMM (cAMM) with predictive capabilities
    Extends CPMM with pseudo-arbitrage and dynamic liquidity adjustment
    """
    
    def __init__(self, liquidity_constant: float, initial_price: float):
        super().__init__(liquidity_constant, initial_price)
        self.liquidity_adjustments = []
        
    def pseudo_arbitrage(self, new_market_price: float) -> Tuple[float, float]:
        """
        Perform pseudo-arbitrage to move curve to new equilibrium
        Following the paper's approach to eliminate divergence
        
        Args:
            new_market_price: New market price from oracle
            
        Returns:
            (adjustment_x, adjustment_y): Required adjustments to reserves
        """
        current_price = self.get_price()
        
        # Calculate new equilibrium point
        # New reserves to match market price while maintaining c
        new_x = np.sqrt(self.c / new_market_price)
        new_y = self.c / new_x
        
        # Calculate adjustments needed
        adjustment_x = new_x - self.x
        adjustment_y = new_y - self.y
        
        # Apply adjustments
        self.x = new_x
        self.y = new_y
        
        # Track adjustments for liquidity provider incentives
        self.liquidity_adjustments.append({
            'adjustment_x': adjustment_x,
            'adjustment_y': adjustment_y,
            'price': new_market_price
        })
        
        return adjustment_x, adjustment_y
    
    def calculate_expected_load(self, current_v: float, 
                               future_v_distribution: np.ndarray,
                               probabilities: np.ndarray) -> float:
        """
        Calculate expected load (divergence * slippage)
        Following Equation 2 from the paper
        
        Args:
            current_v: Current equilibrium valuation
            future_v_distribution: Array of possible future valuations
            probabilities: Probability distribution over future valuations
            
        Returns:
            Expected load
        """
        total_load = 0
        
        for future_v, prob in zip(future_v_distribution, probabilities):
            # Calculate divergence loss for this future state
            div_loss = self._divergence_loss_between_v(current_v, future_v)
            
            # Calculate slippage loss for this future state
            slip_loss = self._slippage_loss_between_v(current_v, future_v)
            
            # Load is product of divergence and slippage
            load = div_loss * slip_loss
            
            # Weight by probability
            total_load += prob * load
        
        return total_load
    
    def _divergence_loss_between_v(self, v1: float, v2: float) -> float:
        """Calculate divergence loss between two valuations"""
        # Following paper's loss_div formula
        phi_v1 = self._phi(v1)
        phi_v2 = self._phi(v2)
        
        loss = v2 * phi_v1[0] + (1 - v2) * phi_v1[1] - \
               (v2 * phi_v2[0] + (1 - v2) * phi_v2[1])
        
        return abs(loss)
    
    def _slippage_loss_between_v(self, v1: float, v2: float) -> float:
        """Calculate slippage loss between two valuations"""
        # Following paper's loss_slip formula
        phi_v1 = self._phi(v1)
        phi_v2 = self._phi(v2)
        
        val_v2 = v2 * phi_v2[0] + (1 - v2) * phi_v2[1]
        val_v1 = v2 * phi_v1[0] + (1 - v2) * phi_v1[1]
        
        loss = ((1 - v2) / (1 - v1)) * (val_v2 - val_v1)
        
        return abs(loss)
    
    def _phi(self, v: float) -> Tuple[float, float]:
        """
        Calculate φ(v) - the equilibrium state for valuation v
        For CPMM: φ(v) = (sqrt((1-v)/v), sqrt(v/(1-v)))
        """
        if v <= 0 or v >= 1:
            raise ValueError("Valuation v must be in (0,1)")
        
        x_eq = np.sqrt(self.c * (1 - v) / v)
        y_eq = np.sqrt(self.c * v / (1 - v))
        
        return x_eq, y_eq
    
    def apply_predictive_liquidity(self, predicted_price: float, 
                                   std_dev: float = None) -> dict:
        """
        Apply Gaussian incentive fee distribution centered on predicted price
        Following the paper's predictive liquidity distribution
        
        Args:
            predicted_price: Predicted future price v'_p
            std_dev: Standard deviation for Gaussian distribution
            
        Returns:
            Dictionary with liquidity distribution info
        """
        if std_dev is None:
            std_dev = config.INCENTIVE_STD_DEV
        
        # Calculate predicted equilibrium valuation
        v_p = 1 / (1 + predicted_price**2)
        
        # Create Gaussian distribution centered on v_p
        # This will guide liquidity provider incentives
        distribution_info = {
            'predicted_price': predicted_price,
            'predicted_valuation': v_p,
            'std_dev': std_dev,
            'center_x': np.sqrt(self.c * (1 - v_p) / v_p),
            'center_y': np.sqrt(self.c * v_p / (1 - v_p))
        }
        
        return distribution_info
    
    def gaussian_incentive_fee(self, position: float, predicted_v: float,
                               std_dev: float) -> float:
        """
        Calculate incentive fee based on Gaussian distribution
        φ(x) = (1/(σ√(2π))) * exp(-0.5*((x-v'_p)/σ)^2)
        
        Args:
            position: Current position/valuation
            predicted_v: Predicted valuation v'_p (mean of distribution)
            std_dev: Standard deviation σ_φ
            
        Returns:
            Incentive fee multiplier
        """
        coefficient = 1 / (std_dev * np.sqrt(2 * np.pi))
        exponent = -0.5 * ((position - predicted_v) / std_dev) ** 2
        
        fee_multiplier = coefficient * np.exp(exponent)
        
        return fee_multiplier
    
    def calculate_liquidity_utilization(self, trade_volume: float) -> float:
        """
        Calculate liquidity utilization metric
        Utilization = Trade Volume / Average Liquidity
        
        Args:
            trade_volume: Total trade volume
            
        Returns:
            Liquidity utilization ratio
        """
        avg_liquidity = np.sqrt(self.c)  # Geometric mean of reserves
        return trade_volume / avg_liquidity
    
    def calculate_liquidity_depth(self, price_impact_threshold: float = 0.01) -> float:
        """
        Calculate liquidity depth - max trade size for acceptable price impact
        
        Args:
            price_impact_threshold: Acceptable price impact (e.g., 0.01 for 1%)
            
        Returns:
            Maximum trade size
        """
        initial_price = self.get_price()
        target_price = initial_price * (1 + price_impact_threshold)
        
        # For buying X: find delta_x such that price increases by threshold
        # New price = c/(x - delta_x)^2
        target_x = np.sqrt(self.c / target_price)
        max_trade_size = self.x - target_x
        
        return max_trade_size


def create_amm(liquidity_constant: float, initial_price: float,
               use_predictive: bool = True):
    """
    Factory function to create AMM instance
    
    Args:
        liquidity_constant: Liquidity constant c
        initial_price: Initial price
        use_predictive: Whether to use configurable virtual AMM
        
    Returns:
        AMM instance
    """
    if use_predictive:
        return ConfigurableVirtualAMM(liquidity_constant, initial_price)
    else:
        return CPMM(liquidity_constant, initial_price)
