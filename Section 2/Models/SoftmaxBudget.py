from Models.BaseModel import *
import numpy as np

class SoftmaxBudget(BaseBudgetedBandit):
    def __init__(self, n_weeks: int, n_users: int, n_arms: int, prices: np.ndarray, budget: float,
                 temperature: float = 1.0, decay_rate: float = 0.995, min_temperature: float = 0.1,
                 initial_q_value: float = 1.0, initial_counts: int = 1, **kwargs):
        super().__init__(n_weeks, n_users, n_arms, prices, budget, **kwargs)
        self.temperature = float(temperature)
        self.decay_rate = float(decay_rate)
        self.min_temperature = float(min_temperature)
        
        # Optimistic Initialization
        self.counts = np.full((self.n_users, self.n_arms), initial_counts, dtype=int)
        self.q_values = np.full((self.n_users, self.n_arms), initial_q_value, dtype=float)

        self.round_num = 0

    def recommend(self) -> np.ndarray:
        self.round_num += 1
        recommendations = np.full(self.n_users, -1, dtype=int)

        current_temperature = max(self.temperature * (self.decay_rate ** self.round_num), self.min_temperature)
        
        # Use mean Q-values for global feasible set selection
        item_scores_for_feasible_set = np.mean(self.q_values, axis=0)
        feasible_items = build_feasible_set_generic(self.prices, self.budget, item_scores_for_feasible_set)

        if not feasible_items:
            self._store_recommendations(recommendations)
            return recommendations

        feasible_items_np = np.array(feasible_items)

        for u in range(self.n_users):
            chosen_arm = -1
            
            # Extract Q-values for only feasible items for this user
            feasible_q_values_for_user = self.q_values[u, feasible_items_np]
            
            # Calculate Softmax probabilities
            max_q = np.max(feasible_q_values_for_user) # For numerical stability
            exp_q_values = np.exp((feasible_q_values_for_user - max_q) / current_temperature)
            sum_exp_q_values = np.sum(exp_q_values)
            
            if sum_exp_q_values > 0:
                probabilities = exp_q_values / sum_exp_q_values
                # Sample an arm based on these probabilities
                chosen_arm_idx_in_feasible = np.random.choice(len(feasible_items_np), p=probabilities)
                chosen_arm = feasible_items_np[chosen_arm_idx_in_feasible]
            elif len(feasible_items_np) > 0:
                # Fallback: if probabilities are all effectively zero, pick randomly from feasible
                chosen_arm = np.random.choice(feasible_items_np)

            recommendations[u] = chosen_arm
        
        self._store_recommendations(recommendations)
        return recommendations

    def update(self, feedback: np.ndarray): # Changed signature
        # Use self.last_recs from BaseBudgetedBandit
        if np.all(self.last_recs == -1):
            return

        for user_idx, arm_idx, reward in zip(range(self.n_users), self.last_recs, feedback):
            if arm_idx == -1: # Skip if no arm was recommended for this user
                continue
            
            self.counts[user_idx, arm_idx] += 1
            self.q_values[user_idx, arm_idx] += (reward - self.q_values[user_idx, arm_idx]) / self.counts[user_idx, arm_idx]

    def _get_current_scores(self) -> np.ndarray:
        """
        Returns the current estimated Q-values for all user-arm pairs.
        """
        # For Softmax, the underlying scores are the Q-values.
        return self.q_values