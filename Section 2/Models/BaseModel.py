import numpy as np
from abc import ABC, abstractmethod

def build_feasible_set_generic(prices: np.ndarray, budget: float, scores: np.ndarray, agg_fn=np.mean, add_noise: bool = False, noise_scale: float = 0.01) -> list:
    """
    Generic greedy feasible set builder using customizable score aggregation
    and score-per-cost prioritization. Can add noise for randomized greedy.

    Args:
        prices (np.ndarray): Cost per item (K,).
        budget (float): Total budget.
        scores (np.ndarray): Score matrix (N, K) or vector (K,).
        agg_fn (callable): Aggregation function over users (axis=0), e.g., np.mean, np.max.
        add_noise (bool): If True, adds Gaussian noise to score_per_cost before sorting.
        noise_scale (float): Standard deviation of the Gaussian noise.

    Returns:
        list: Selected item indices within budget.
    """
    if scores.ndim == 2:
        item_scores = agg_fn(scores, axis=0)
    else:
        item_scores = scores

    score_per_cost = item_scores / (prices + 1e-8) # Add a small epsilon to prices to avoid division by zero.

    if add_noise:
        # Add Gaussian noise to introduce randomness in the greedy selection
        noise = np.random.normal(0, noise_scale, size=score_per_cost.shape)
        score_per_cost += noise

    idx = np.lexsort((np.arange(len(prices)), prices, -score_per_cost)) 

    S, total_cost = [], 0.0
    for j in idx:
        if total_cost + prices[j] <= budget:
            S.append(j)
            total_cost += prices[j]
    return S



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


