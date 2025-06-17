import numpy as np
from abc import ABC, abstractmethod

def build_feasible_set_generic(prices: np.ndarray, budget: float, scores: np.ndarray, agg_fn=np.mean, add_noise: bool = True, noise_scale: float = 1e-4) -> list:
    """
    Modified greedy feasible set builder based on marginal gain.
    Selects items by prioritizing those that offer the largest
    additional total expected value across all users per unit cost.
    Can optionally add noise to break ties or introduce randomness in selection.

    Args:
        prices (np.ndarray): Cost per item (K,).
        budget (float): Total budget.
        scores (np.ndarray): UCB score matrix (N_users, K_arms).
        agg_fn (callable): Placeholder for compatibility, not directly used in this logic.
        add_noise (bool): If True, adds Gaussian noise to the marginal gain per cost.
        noise_scale (float): Standard deviation of the Gaussian noise.

    Returns:
        list: Selected item indices within budget.
    """
    n_users, n_arms = scores.shape
    
    purchased_arms = []
    current_cost = 0.0

    current_best_ucb_per_user = np.zeros(n_users) 
    available_arms = set(range(n_arms))

    while True:
        best_arm_to_add = -1
        max_marginal_gain_per_cost = -np.inf # Initialize with negative infinity
        
        candidates = [] # Store (marginal_gain_per_cost, arm_idx) for potential noise application
        
        for arm_idx in list(available_arms):
            if current_cost + prices[arm_idx] <= budget:
                potential_new_best_ucb_per_user = np.maximum(current_best_ucb_per_user, scores[:, arm_idx])
                marginal_gain = np.sum(potential_new_best_ucb_per_user) - np.sum(current_best_ucb_per_user)
                
                if prices[arm_idx] > 1e-8: # Use a small epsilon to avoid division by near-zero prices
                    mg_per_cost = marginal_gain / prices[arm_idx]
                elif marginal_gain > 0:
                    mg_per_cost = np.inf # Free arm with positive gain is infinitely good
                else:
                    mg_per_cost = 0.0 # Free arm with no gain

                candidates.append((mg_per_cost, arm_idx))

        if not candidates: # No more arms can be added within budget
            break

        # Apply noise if requested
        if add_noise:
            # Convert candidates to an array for vectorized noise addition
            # Format: [(mg_per_cost_0, arm_idx_0), (mg_per_cost_1, arm_idx_1), ...]
            candidate_gains = np.array([c[0] for c in candidates])
            noise = np.random.normal(0, noise_scale, size=len(candidates))
            candidate_gains += noise
            
            # Find the best arm based on noisy gains
            best_candidate_idx = np.argmax(candidate_gains)
            best_arm_to_add = candidates[best_candidate_idx][1] # Get original arm_idx
            max_marginal_gain_per_cost = candidate_gains[best_candidate_idx] # Get the noisy gain
        else:
            # Find the best arm based on original gains
            max_marginal_gain_per_cost_unnoisy = -np.inf
            for mg_per_cost, arm_idx in candidates:
                if mg_per_cost > max_marginal_gain_per_cost_unnoisy:
                    max_marginal_gain_per_cost_unnoisy = mg_per_cost
                    best_arm_to_add = arm_idx
            max_marginal_gain_per_cost = max_marginal_gain_per_cost_unnoisy


        if best_arm_to_add == -1 or max_marginal_gain_per_cost <= 0:
            break # No more beneficial arms can be added or budget is exhausted
        
        # Add the chosen arm
        purchased_arms.append(best_arm_to_add)
        current_cost += prices[best_arm_to_add]
        available_arms.remove(best_arm_to_add)
        
        # Update current_best_ucb_per_user for the next iteration
        current_best_ucb_per_user = np.maximum(current_best_ucb_per_user, scores[:, best_arm_to_add])
        
    return purchased_arms



class BaseBudgetedBandit(ABC):
    """
    Abstract base class for combinatorial budgeted bandit models.
    Defines the common interface and shared attributes for all bandit agents.
    """
    def __init__(self, n_weeks: int, n_users: int, n_arms: int, prices: np.ndarray, budget: float, **kwargs):
        self.n_weeks = n_weeks
        self.n_users = n_users
        self.n_arms = n_arms
        self.prices = prices
        self.budget = budget
        
        # Common state variables for tracking recommendations and rewards
        self.last_recs = np.full(self.n_users, -1, dtype=int)

        self.current_week = 0

    @abstractmethod
    def recommend(self) -> np.ndarray:
        """
        Abstract method to generate recommendations for the current round.
        Must be implemented by all concrete bandit models.
        """
        pass

    @abstractmethod
    def update(self, feedback: np.ndarray):
        """
        Abstract method to update the bandit model based on observed feedback.
        Must be implemented by all concrete bandit models.
        """
        pass

    @abstractmethod
    def _get_current_scores(self) -> np.ndarray:
        """
        Abstract method to return the current estimated scores/probabilities/UCB values
        for all user-arm pairs. This is primarily for ensemble methods.
        Must be implemented by all concrete bandit models.
        Returns a (n_users, n_arms) array.
        """
        pass
    
    def _store_recommendations(self, recommendations: np.ndarray):
        """Helper method to store the last set of recommendations."""
        self.last_recs = recommendations


