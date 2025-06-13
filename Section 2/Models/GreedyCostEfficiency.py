from Models.BaseModel import *
import numpy as np

class GreedyCostEfficiency(BaseBudgetedBandit):
    def __init__(self, n_weeks: int, n_users: int, n_arms: int, prices: np.ndarray, budget: float, explore_rounds: int = 50, initial_q_value: float = 0.5, initial_counts: int = 1, exploitation_noise_scale: float = 0.001, **kwargs):
        super().__init__(n_weeks, n_users, n_arms, prices, budget, **kwargs)
        self.explore_rounds = explore_rounds
        self.current_round = 0
        self.exploitation_noise_scale = exploitation_noise_scale 

        # User-specific counts and rewards
        self.arm_counts = np.full((n_users, n_arms), initial_counts, dtype=float)
        self.arm_rewards = np.full((n_users, n_arms), initial_q_value * initial_counts, dtype=float)
        self.q_values = np.full((n_users, n_arms), initial_q_value, dtype=float)

    def recommend(self) -> np.ndarray:
        self.current_round += 1
        recommended_arms_for_users = np.full(self.n_users, -1, dtype=int)

        if self.current_round <= self.explore_rounds:
            # Exploration phase: Select a feasible set randomly (via noise in score_per_cost)
            feasible_set_S = build_feasible_set_generic(self.prices, self.budget, self.q_values, agg_fn=np.mean, add_noise=True, noise_scale=1.0) # High noise for truly random
        else:
            # Exploitation phase: Select a feasible set based on aggregated Q-values, with slight randomization
            feasible_set_S = build_feasible_set_generic(self.prices, self.budget, self.q_values, agg_fn=np.mean, add_noise=True, noise_scale=self.exploitation_noise_scale) # Controlled noise

        if not feasible_set_S:
            self._store_recommendations(recommended_arms_for_users)
            return recommended_arms_for_users
            
        for u in range(self.n_users):
            if feasible_set_S:
                chosen_arm = max(feasible_set_S, key=lambda a: self.q_values[u, a])
                recommended_arms_for_users[u] = chosen_arm
        
        self._store_recommendations(recommended_arms_for_users)
        return recommended_arms_for_users

    def update(self, results: np.array):
        # Use self.last_recs inherited from BaseBudgetedBandit
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
        return self.q_values