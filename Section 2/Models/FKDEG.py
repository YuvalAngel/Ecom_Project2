from Models.BaseModel import *
import numpy as np

class FractionalKnapsackDecreasingEpsilonGreedy(BaseBudgetedBandit):
    def __init__(self, n_weeks: int, n_users: int, n_arms: int, prices: np.ndarray, budget: float, initial_epsilon: float = 1.0, epsilon_decay: float = 0.99, min_epsilon: float = 0.05, initial_q_value: float = 0.5, initial_counts: int = 1, **kwargs):
        super().__init__(n_weeks, n_users, n_arms, prices, budget, **kwargs)
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # User-specific counts and rewards
        self.arm_counts = np.full((n_users, n_arms), initial_counts, dtype=float)
        self.arm_rewards = np.full((n_users, n_arms), initial_q_value * initial_counts, dtype=float)
        self.q_values = np.full((n_users, n_arms), initial_q_value, dtype=float)

    def recommend(self) -> np.ndarray:
        recommended_arms_for_users = np.full(self.n_users, -1, dtype=int)

        if np.random.rand() < self.epsilon:
            # Exploration: Select a feasible set randomly
            feasible_set_S = build_feasible_set_generic(self.prices, self.budget, self.q_values, agg_fn=np.mean, add_noise=True, noise_scale=1.0)
        else:
            # Exploitation: Select a feasible set based on aggregated Q-values
            feasible_set_S = build_feasible_set_generic(self.prices, self.budget, self.q_values, agg_fn=np.mean)

        if not feasible_set_S:
            self._store_recommendations(recommended_arms_for_users)
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            return recommended_arms_for_users
            
        for u in range(self.n_users):
            chosen_arm = max(feasible_set_S, key=lambda a: self.q_values[u, a])
            recommended_arms_for_users[u] = chosen_arm
        
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
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
