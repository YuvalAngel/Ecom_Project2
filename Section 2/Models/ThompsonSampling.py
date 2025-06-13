from Models.BaseModel import *
import numpy as np

class ThompsonSampling(BaseBudgetedBandit):
    def __init__(self, n_weeks: int, n_users: int, n_arms: int, prices: np.ndarray, budget: float, 
                 alpha_prior: float = 1.0, beta_prior: float = 1.0, **kwargs):
        super().__init__(n_weeks, n_users, n_arms, prices, budget, **kwargs) # Pass n_weeks and kwargs to base
        self.successes = np.zeros((self.n_users, self.n_arms), dtype=float)
        self.failures = np.zeros((self.n_users, self.n_arms), dtype=float)
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior 
        self.last_sampled_probabilities = None # To store for _get_current_scores

    def recommend(self) -> np.ndarray:
        # Sample click-through probabilities from the Beta posterior for each user-arm pair.
        # The Beta distribution parameters are (actual_successes + prior_alpha) and (actual_failures + prior_beta).
        samples = np.random.beta(self.successes + self.alpha_prior, self.failures + self.beta_prior)
        self.last_sampled_probabilities = samples # Store for _get_current_scores

        # Build the feasible set using the average of sampled probabilities across users.
        feasible_set_S = build_feasible_set_generic(self.prices, self.budget, samples, agg_fn=np.mean)

        recommendations = np.full(self.n_users, -1, dtype=int)
        if not feasible_set_S:
            self._store_recommendations(recommendations)
            return recommendations
        else:
            for u in range(self.n_users):
                # For each user, choose the arm from the feasible set S that has the maximum sampled value for them.
                chosen_arm = max(feasible_set_S, key=lambda a: samples[u, a])
                recommendations[u] = chosen_arm
        
        self._store_recommendations(recommendations)
        return recommendations

    def update(self, results: np.ndarray): # Changed signature to match BaseBudgetedBandit
        # Use self.last_recs from BaseBudgetedBandit
        if not hasattr(self, 'last_recs') or np.all(self.last_recs == -1):
            return

        # 'results' contains rewards corresponding to 'self.last_recs'
        for user_idx, arm_idx, reward in zip(range(self.n_users), self.last_recs, results):
            if arm_idx == -1: # Skip if no arm was recommended for this user
                continue
            
            if reward > 0: # Reward is 1 (success)
                self.successes[user_idx, arm_idx] += 1
            else: # Reward is 0 (failure)
                self.failures[user_idx, arm_idx] += 1
    
    def _get_current_scores(self) -> np.ndarray:
        """
        Returns the mean of the Beta posterior distributions for all user-arm pairs.
        """
        # Calculate mean of Beta distribution: alpha / (alpha + beta)
        # Add a small epsilon to the denominator to prevent division by zero if alpha+beta is 0
        return (self.successes + self.alpha_prior) / (self.successes + self.failures + self.alpha_prior + self.beta_prior + 1e-8)
