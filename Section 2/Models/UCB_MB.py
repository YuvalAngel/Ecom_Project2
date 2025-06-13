from Models.BaseModel import *
import numpy as np

class UCB_MB(BaseBudgetedBandit):
    def __init__(self, n_weeks: int, n_users: int, n_arms: int, prices: np.ndarray, budget: float, c: float = 0.1, initial_q_value: float = 1.0, initial_counts: int = 1, **kwargs):
        super().__init__(n_weeks, n_users, n_arms, prices, budget, **kwargs)
        self.c = c # Exploration parameter
        self.t = 0  # Global time step (number of rounds played)

        # User-specific counts and rewards (for mean reward calculation)
        self.arm_counts = np.full((n_users, n_arms), initial_counts, dtype=float)
        # Store sum of rewards for each user-arm pair
        self.arm_rewards = np.full((n_users, n_arms), initial_q_value * initial_counts, dtype=float)
        
        # User-specific Q-values (mean rewards)
        self.q_values = np.full((n_users, n_arms), initial_q_value, dtype=float)

    def _calculate_ucb_scores(self) -> np.ndarray:
        """Helper to calculate UCB values based on current state."""
        ucb_scores = np.zeros((self.n_users, self.n_arms))
        # Ensure log_t is not zero or negative for small t
        log_t = np.log(self.t + 1) # Add 1 to t to ensure log(t) is always >= log(1)=0

        # Add a small epsilon to counts to avoid division by zero
        counts_safe = self.arm_counts + 1e-8
        
        # Empirical variance for Bernoulli rewards: p * (1 - p)
        # Use np.clip to ensure q_values are within [0, 1] for variance calculation stability.
        clipped_q_values = np.clip(self.q_values, 0, 1)
        empirical_variance = clipped_q_values * (1 - clipped_q_values)
        
        # UCB formula: mean + c * sqrt(2 * variance * ln(T) / n_i)
        ucb_raw = (
            self.q_values  # Mean reward
            + self.c * np.sqrt(2 * empirical_variance * log_t / counts_safe) 
        )
        MAX_UCB_CAP = 5.0 # A configurable parameter or a heuristic
        ucb_capped = np.clip(ucb_raw, None, MAX_UCB_CAP)

        ucb_capped[self.arm_counts == 0] = MAX_UCB_CAP # Ensure unplayed arms get the highest possible score

        return ucb_capped

    def recommend(self) -> np.ndarray:
        self.t += 1
        recommended_arms_for_users = np.full(self.n_users, -1, dtype=int)

        ucb_values_for_round = self._calculate_ucb_scores() 

        feasible_set_S = build_feasible_set_generic(self.prices, self.budget, ucb_values_for_round, agg_fn=np.mean)

        if not feasible_set_S:
            self._store_recommendations(recommended_arms_for_users)
            return recommended_arms_for_users
            
        for u in range(self.n_users):
            if feasible_set_S:
                chosen_arm = max(feasible_set_S, key=lambda a: ucb_values_for_round[u, a])
                recommended_arms_for_users[u] = chosen_arm
        
        self._store_recommendations(recommended_arms_for_users)
        return recommended_arms_for_users

    def update(self, results: np.array):
        # We need to make sure _last_recommended_arms is stored via super()._store_recommendations()
        # This check should ideally be done in _store_recommendations if needed, or assume it's always set.
        if not hasattr(self, 'last_recs') or np.all(self.last_recs == -1): # Check if any valid recs exist
             return

        for user_idx, arm_idx in enumerate(self.last_recs): # Use self.last_recs
            if arm_idx == -1:
                continue
            reward = results[user_idx]
            self.arm_counts[user_idx, arm_idx] += 1
            self.arm_rewards[user_idx, arm_idx] += reward
            self.q_values[user_idx, arm_idx] = self.arm_rewards[user_idx, arm_idx] / self.arm_counts[user_idx, arm_idx]
    
    def _get_current_scores(self) -> np.ndarray:
        return self._calculate_ucb_scores()