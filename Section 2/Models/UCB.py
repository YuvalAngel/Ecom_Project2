from Models.BaseModel import *
import numpy as np

class UCB(BaseBudgetedBandit):
    def __init__(self, n_weeks: int, n_users: int, n_arms: int, prices: np.ndarray, budget: float, 
                 c: float = 1.0, initial_q_value: float = 1.0, initial_counts: float = 1.0, **kwargs): # Added initial_q_value and initial_counts for optimistic init
        super().__init__(n_weeks, n_users, n_arms, prices, budget, **kwargs)
        self.c = c 
        self.t = 0 # Global time step (number of rounds played)

        # Optimistic Initialization
        self.counts = np.full((self.n_users, self.n_arms), initial_counts) 
        self.values = np.full((self.n_users, self.n_arms), initial_q_value) 
        
        # The similarity matrix is retained from your original code, if intended for UCB calculation.
        # This is a non-standard UCB addition.
        self.similarity = np.eye(n_users) * 0.8 + 0.2 / n_users 
        
    def _calculate_ucb_scores(self) -> np.ndarray:
        """Helper to calculate UCB values based on current state."""
        # Ensure log_t is not zero or negative for small t
        log_t = np.log(self.t + 1) # Add 1 to t to ensure log(t) is always >= log(1)=0

        # Add a small epsilon to counts to avoid division by zero
        counts_safe = self.counts + 1e-8 

        # Empirical variance for Bernoulli rewards: p * (1 - p)
        clipped_values = np.clip(self.values, 0, 1)
        empirical_variance = clipped_values * (1 - clipped_values)
        
        # UCB formula: mean + c * sqrt(2 * variance * ln(T) / n_i)
        ucb_raw = (
            self.values 
            + self.c * np.sqrt(2 * empirical_variance * log_t / counts_safe) 
        )
        
        # Apply similarity smoothing and normalization if similarity matrix is used
        if hasattr(self, 'similarity'):
            # The similarity matrix sums to 1 across rows (normalized for averaging)
            ucb_smoothed = (self.similarity @ ucb_raw) / self.similarity.sum(axis=1, keepdims=True)
        else:
            ucb_smoothed = ucb_raw # If no similarity, just use raw UCB
        
        # Cap UCB scores to a reasonable maximum before returning
        MAX_UCB_CAP = 5.0 # Use the same or similar cap as UCB_MB
        ucb_capped = np.clip(ucb_smoothed, None, MAX_UCB_CAP)

        # Ensure truly unexplored arms (if initial_counts was 0) get the max cap
        ucb_capped[self.counts == 0] = MAX_UCB_CAP

        return ucb_capped

    def recommend(self) -> np.ndarray:
        self.t += 1 # Increment global round counter
        recommended_arms_for_users = np.full(self.n_users, -1, dtype=int)

        ucb_values_for_round = self._calculate_ucb_scores() 

        # Build the feasible set using build_feasible_set_generic, averaging UCB scores.
        feasible_set_S = build_feasible_set_generic(self.prices, self.budget, ucb_values_for_round, agg_fn=np.mean)

        if not feasible_set_S:
            self._store_recommendations(recommended_arms_for_users)
            return recommended_arms_for_users
            
        for u in range(self.n_users):
            if feasible_set_S:
                # For each user, choose the arm from the feasible set S that has the maximum UCB value for them.
                chosen_arm = max(feasible_set_S, key=lambda a: ucb_values_for_round[u, a])
                recommended_arms_for_users[u] = chosen_arm
        
        self._store_recommendations(recommended_arms_for_users)
        return recommended_arms_for_users

    def update(self, results: np.ndarray): # Changed signature to match BaseBudgetedBandit
        # Use self.last_recs from BaseBudgetedBandit
        if not hasattr(self, 'last_recs') or np.all(self.last_recs == -1):
            return

        for user_idx, arm_idx, reward in zip(range(self.n_users), self.last_recs, results):
            if arm_idx == -1:
                continue
            
            # Update counts
            self.counts[user_idx, arm_idx] += 1
            n = self.counts[user_idx, arm_idx]
            
            # Update mean reward for Bernoulli rewards using incremental average
            v_old = self.values[user_idx, arm_idx]
            self.values[user_idx, arm_idx] = ((n - 1) / n) * v_old + (1 / n) * reward
    
    def _get_current_scores(self) -> np.ndarray:
        """
        Returns the current UCB scores for all user-arm pairs.
        """
        return self._calculate_ucb_scores()