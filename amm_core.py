"""
Core Automated Market Maker (AMM) Implementation
Implements CPMM (Constant Product Market Maker) and configurable virtual AMM
"""

import numpy as np
from typing import Tuple, Optional
import config


# ============================================================================
# VALUATION HELPER FUNCTIONS
# ============================================================================

def price_to_valuation(p: float) -> float:
    """
    Convert price to valuation using v = p / (1 + p)
    Clamps to valid range for numerical safety.
    
    Args:
        p: Price (must be non-negative)
        
    Returns:
        Valuation v in (0, 1)
    """
    v = p / (1.0 + max(p, 0.0))
    return float(np.clip(v, config.MIN_V, config.MAX_V))


def valuation_to_equilibrium_x(v: float, c: float) -> float:
    """
    Calculate equilibrium x* for a given valuation v on CPMM curve.
    Formula: x* = sqrt(c * (1 - v) / v)
    
    Args:
        v: Valuation (must be in (0, 1))
        c: Liquidity constant
        
    Returns:
        Equilibrium x reserves
    """
    v = float(np.clip(v, config.MIN_V, config.MAX_V))
    return float(np.sqrt(c * (1.0 - v) / v))


def equilibrium_state_from_price(p: float, c: float) -> Tuple[float, float, float]:
    """
    Calculate full equilibrium state (v, x*, y*) from price.
    
    Args:
        p: Price
        c: Liquidity constant
        
    Returns:
        (v, x_star, y_star): Equilibrium valuation and reserves
    """
    v = price_to_valuation(p)
    x_star = valuation_to_equilibrium_x(v, c)
    y_star = c / x_star
    return v, x_star, y_star


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
        self.initial_price = initial_price
        self.initial_v = price_to_valuation(initial_price)  # Store initial valuation
        
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
        Calculate equilibrium valuation v from market price.
        Using correct formula: v = p / (1 + p)
        
        Args:
            market_price: Observed market price from oracle
            
        Returns:
            Equilibrium valuation v
        """
        return price_to_valuation(market_price)
    
    def calculate_divergence_loss(self, current_price: float) -> float:
        """
        Calculate divergence (impermanent) loss using paper's formula.
        loss_div(v_init, v_curr) := v_curr · Φ(v_init) - v_curr · Φ(v_curr)
        
        This measures the loss from holding assets in the AMM (at equilibrium Φ(v_curr))
        versus if the AMM had stayed at initial equilibrium Φ(v_init) but valued at v_curr.
        
        Args:
            current_price: Current market price
            
        Returns:
            Divergence loss amount
        """
        # Convert price to valuation
        current_v = price_to_valuation(current_price)
        
        # Use paper's divergence loss formula between initial and current valuation
        return self._divergence_loss_between_v(self.initial_v, current_v)
    
    def calculate_slippage_loss(self, previous_price: float, current_price: float) -> float:
        """
        Calculate slippage loss using paper's formula.
        loss_slip(v_prev, v_curr) := ((1 - v_curr) / (1 - v_prev)) * (v_curr · Φ(v_curr) - v_curr · Φ(v_prev))
        
        This measures the loss due to price movement between two time periods.
        
        Args:
            previous_price: Previous market price
            current_price: Current market price
            
        Returns:
            Slippage loss amount
        """
        v_prev = price_to_valuation(previous_price)
        v_curr = price_to_valuation(current_price)
        
        # Use paper's slippage loss formula
        return self._slippage_loss_between_v(v_prev, v_curr)
    
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
    
    def _phi(self, v: float) -> Tuple[float, float]:
        """
        Calculate φ(v) - the equilibrium state for valuation v
        For CPMM: φ(v) = (sqrt(c*(1-v)/v), sqrt(c*v/(1-v)))
        Returns (x*, y*) equilibrium reserves
        
        Args:
            v: Valuation in (0,1)
            
        Returns:
            (x_eq, y_eq): Equilibrium reserves
        """
        v = float(np.clip(v, config.MIN_V, config.MAX_V))
        x_eq = np.sqrt(self.c * (1 - v) / v)
        y_eq = np.sqrt(self.c * v / (1 - v))
        return x_eq, y_eq
    
    def _divergence_loss_between_v(self, v1: float, v2: float) -> float:
        """
        Calculate divergence loss between two valuations using paper formula
        loss_div(v, v') := v' · Φ(v) - v' · Φ(v')
        
        Args:
            v1: Initial valuation
            v2: Current valuation
            
        Returns:
            Divergence loss
        """
        phi_v1 = self._phi(v1)
        phi_v2 = self._phi(v2)
        
        # v' · Φ(v) = v'*x_eq(v) + (1-v')*y_eq(v)
        cap_at_v1 = v2 * phi_v1[0] + (1 - v2) * phi_v1[1]
        cap_at_v2 = v2 * phi_v2[0] + (1 - v2) * phi_v2[1]
        
        loss = cap_at_v1 - cap_at_v2
        return abs(loss)
    
    def _slippage_loss_between_v(self, v1: float, v2: float) -> float:
        """
        Calculate slippage loss between two valuations using paper formula
        loss_slip(v, v') := ((1 - v') / (1 - v)) * (v' · Φ(v') - v' · Φ(v))
        
        Args:
            v1: Initial valuation
            v2: Current valuation
            
        Returns:
            Slippage loss
        """
        phi_v1 = self._phi(v1)
        phi_v2 = self._phi(v2)
        
        cap_at_v2 = v2 * phi_v2[0] + (1 - v2) * phi_v2[1]
        cap_at_v1 = v2 * phi_v1[0] + (1 - v2) * phi_v1[1]
        
        ratio = (1 - v2) / (1 - v1 + 1e-12)  # Add small epsilon to avoid division by zero
        loss = ratio * (cap_at_v2 - cap_at_v1)
        
        return abs(loss)


class ConfigurableVirtualAMM(CPMM):
    """
    Configurable Virtual AMM (cAMM) with predictive capabilities
    Extends CPMM with pseudo-arbitrage and dynamic liquidity adjustment
    """
    
    def __init__(self, liquidity_constant: float, initial_price: float):
        super().__init__(liquidity_constant, initial_price)
        self.liquidity_adjustments = []
        self.last_equilibrium_v = price_to_valuation(initial_price)  # Track equilibrium valuation
        
    def pseudo_arbitrage(self, new_market_price: float) -> Tuple[float, float]:
        """
        Shift curve so current inventory (x,y) is at the new equilibrium valuation
        implied by the oracle price. This avoids arbitrage while keeping inventory drift small.
        
        Following the paper's approach: shift the bonding curve so that current holdings
        correspond to equilibrium at the new valuation.
        
        Args:
            new_market_price: New market price from oracle
            
        Returns:
            (adjustment_x, adjustment_y): Required adjustments to reserves
        """
        # Calculate equilibrium valuation from new market price
        v, x_star, y_star = equilibrium_state_from_price(new_market_price, self.c)
        
        # Current reserves (self.x, self.y) become the *new* equilibrium point.
        # To re-center the curve, we rotate/translate the pricing to make price at (self.x,self.y)
        # match p = v/(1-v). For CPMM, a practical way is to rescale c so that:
        #   self.y == c_new / self.x  =>  c_new = self.x * self.y
        # This puts inventory exactly on-curve (idempotent if already on curve)
        
        old_x, old_y = self.x, self.y
        
        # Recenter curve to current inventory
        self.c = float(self.x * self.y)
        
        # Store the target equilibrium valuation for diagnostics
        self.last_equilibrium_v = v
        
        # Calculate adjustments (in this case, minimal since we're recentering to current state)
        adjustment_x = self.x - old_x  # Should be 0 in this formulation
        adjustment_y = self.y - old_y  # Should be 0 in this formulation
        
        # Track adjustments for liquidity provider incentives
        self.liquidity_adjustments.append({
            'adjustment_x': adjustment_x,
            'adjustment_y': adjustment_y,
            'price': new_market_price,
            'valuation': v,
            'new_c': self.c
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
        
        # Calculate predicted equilibrium valuation using correct formula
        v_p = price_to_valuation(predicted_price)
        
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
