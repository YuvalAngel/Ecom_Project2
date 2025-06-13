from Models.BaseModel import *
import numpy as np

class GreedyBudget(BaseBudgetedBandit):
    def __init__(self, n_weeks: int, n_users: int, n_arms: int, prices: np.ndarray, budget: float,
                 initial_q_value: float = 1.0, initial_counts: int = 1, noise_scale: float = 0.05, **kwargs):
        super().__init__(n_weeks, n_users, n_arms, prices, budget, **kwargs) # Pass n_weeks and kwargs to base
        self.noise_scale = noise_scale # Noise for recommendation, but not strictly for internal Q-value updates
        
        # Optimistic Initialization for Q-values and counts
        self.counts = np.full((self.n_users, self.n_arms), initial_counts, dtype=int)
        self.q_values = np.full((self.n_users, self.n_arms), initial_q_value, dtype=float)

    def recommend(self) -> np.ndarray:
        recommendations = np.full(self.n_users, -1, dtype=int)

        item_scores_for_feasible_set = np.mean(self.q_values, axis=0) # Aggregate Q-values for overall item score

        feasible_items = build_feasible_set_generic(self.prices, self.budget, item_scores_for_feasible_set)

        if not feasible_items:
            self._store_recommendations(recommendations)
            return recommendations

        feasible_items_np = np.array(feasible_items)

        for u in range(self.n_users):
            chosen_arm = -1
            # For each user, select the arm from the feasible set with the highest Q-value
            best_individual_q_value = -np.inf 
            for arm_idx in feasible_items_np:
                current_item_q_value = self.q_values[u, arm_idx]
                if current_item_q_value > best_individual_q_value:
                    best_individual_q_value = current_item_q_value
                    chosen_arm = arm_idx
            
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
            
            self.counts[user_idx, arm_idx] += 1
            # Standard Q-value update (running average)
            self.q_values[user_idx, arm_idx] += (reward - self.q_values[user_idx, arm_idx]) / self.counts[user_idx, arm_idx]
    
    def _get_current_scores(self) -> np.ndarray:
        """
        Returns the current estimated Q-values for all user-arm pairs.
        """
        return self.q_values
