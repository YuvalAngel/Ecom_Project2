from Models.BaseModel import *
import numpy as np

class AdaptiveBudgetCombinatorialBandit(BaseBudgetedBandit):
    def __init__(self, n_weeks: int, n_users: int, n_arms: int, prices: np.ndarray, budget: float, alpha: float = 0.5, initial_q_value: float = 1.0, initial_counts: int = 1, **kwargs):
        super().__init__(n_weeks, n_users, n_arms, prices, budget, **kwargs)
        self.alpha = alpha # Exploration parameter for UCB-like component
        self.t = 0 # Global time step

        # User-specific counts and rewards
        self.arm_counts = np.full((n_users, n_arms), initial_counts, dtype=float)
        self.arm_rewards = np.full((n_users, n_arms), initial_q_value * initial_counts, dtype=float)
        self.q_values = np.full((n_users, n_arms), initial_q_value, dtype=float)
        
    def _calculate_estimated_values(self) -> np.ndarray:
        """Helper to calculate estimated values for each user and arm, including exploration bonus."""
        estimated_values = np.zeros((self.n_users, self.n_arms))
        # Ensure log_t is not zero or negative for small t. Use self.t + 1 for log_t.
        log_t = np.log(self.t + 1) # Add 1 to t to ensure log(t) is always >= log(1)=0

        # Add a small epsilon to counts to avoid division by zero
        arm_counts_safe = self.arm_counts + 1e-8

        # Empirical variance for Bernoulli rewards: p * (1 - p)
        # Use np.clip to ensure q_values are within [0, 1] for variance calculation stability.
        clipped_q_values = np.clip(self.q_values, 0, 1)
        empirical_variance = clipped_q_values * (1 - clipped_q_values)
        
        # Calculate raw estimated values (mean + exploration bonus)
        estimated_values_raw = (
            self.q_values 
            + self.alpha * np.sqrt(2 * empirical_variance * log_t / arm_counts_safe)
        )

        # Cap estimated values to a reasonable maximum. 
        # Use the same MAX_UCB_CAP for consistency with other UCB-like models.
        MAX_ESTIMATED_VALUE_CAP = 5.0 
        estimated_values_capped = np.clip(estimated_values_raw, None, MAX_ESTIMATED_VALUE_CAP)

        # For truly unplayed arms (if initial_counts was 0 and an arm hasn't been chosen yet),
        # their score should be highest to guarantee exploration.
        # If `initial_counts` is > 0, this check might not be strictly necessary,
        # but it's good practice for robustness.
        estimated_values_capped[self.arm_counts == 0] = MAX_ESTIMATED_VALUE_CAP # Ensure unplayed arms get the highest possible score

        return estimated_values_capped

    def recommend(self) -> np.ndarray:
        self.t += 1
        recommended_arms_for_users = np.full(self.n_users, -1, dtype=int)

        estimated_values_per_user = self._calculate_estimated_values()

        feasible_set_S = build_feasible_set_generic(self.prices, self.budget, estimated_values_per_user, agg_fn=np.mean)

        if not feasible_set_S:
            self._store_recommendations(recommended_arms_for_users)
            return recommended_arms_for_users
            
        for u in range(self.n_users):
            if feasible_set_S: 
                chosen_arm = max(feasible_set_S, key=lambda a: estimated_values_per_user[u, a])
                recommended_arms_for_users[u] = chosen_arm
        
        self._store_recommendations(recommended_arms_for_users)
        return recommended_arms_for_users

    def update(self, results: np.array):
        # Use self.last_recs as it's inherited from BaseBudgetedBandit
        if not hasattr(self, 'last_recs') or np.all(self.last_recs == -1):
            return

        for user_idx, arm_idx in enumerate(self.last_recs):
            if arm_idx == -1:
                continue
            reward = results[user_idx]
            self.arm_counts[user_idx, arm_idx] += 1
            self.arm_rewards[user_idx, arm_idx] += reward
            self.q_values[user_idx, arm_idx] = self.arm_rewards[user_idx, arm_idx] / self.arm_counts[user_idx, arm_idx]
    
    def _get_current_scores(self) -> np.ndarray:
        return self._calculate_estimated_values()