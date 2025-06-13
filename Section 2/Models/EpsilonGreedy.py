from Models.BaseModel import *
import numpy as np

class EpsilonGreedy(BaseBudgetedBandit):
    def __init__(self, n_weeks: int, n_users: int, n_arms: int, prices: np.ndarray, budget: float, epsilon: float, decay: float, min_epsilon: float = 0.05,
                 initial_q_value: float = 1.0, initial_counts: int = 1, **kwargs):
        super().__init__(n_weeks, n_users, n_arms, prices, budget, **kwargs) # Pass n_weeks and kwargs to base
        self.epsilon = float(epsilon)
        self.decay = float(decay)
        self.min_epsilon = float(min_epsilon)
        
        # Optimistic Initialization
        self.counts = np.full((self.n_users, self.n_arms), initial_counts, dtype=int)
        self.q_values = np.full((self.n_users, self.n_arms), initial_q_value, dtype=float)

        self.round_num = 0

    def recommend(self) -> np.ndarray:
        self.round_num += 1
        recommendations = np.full(self.n_users, -1, dtype=int)

        # The core scores for decision-making are the Q-values
        estimated_values_per_user_item = self.q_values.copy()

        # Build feasible set based on average Q-values of items
        item_scores_for_feasible_set = np.mean(estimated_values_per_user_item, axis=0)
        feasible_items = build_feasible_set_generic(self.prices, self.budget, item_scores_for_feasible_set)

        if not feasible_items:
            self._store_recommendations(recommendations)
            return recommendations

        feasible_items_np = np.array(feasible_items)

        current_epsilon = max(self.epsilon * (self.decay ** self.round_num), self.min_epsilon)

        for u in range(self.n_users):
            chosen_arm = -1
            
            if np.random.rand() < current_epsilon:
                # Explore: choose a random arm from the feasible set
                if len(feasible_items_np) > 0:
                    chosen_arm = np.random.choice(feasible_items_np)
            else:
                # Exploit: choose the best arm from the feasible set based on Q-values
                best_individual_q_value = -np.inf
                for arm_idx in feasible_items_np:
                    current_item_q_value = estimated_values_per_user_item[u, arm_idx]
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
            # Update Q-value using incremental mean
            self.q_values[user_idx, arm_idx] += (reward - self.q_values[user_idx, arm_idx]) / self.counts[user_idx, arm_idx]

    def _get_current_scores(self) -> np.ndarray:
        """
        Returns the current estimated Q-values for all user-arm pairs.
        """
        return self.q_values