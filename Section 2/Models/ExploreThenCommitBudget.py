from Models.BaseModel import *
import numpy as np

class ExploreThenCommitBudget(BaseBudgetedBandit):
    def __init__(self, n_weeks: int, n_users: int, n_arms: int, prices: np.ndarray, budget: float,
                 explore_rounds: int, initial_q_value: float = 1.0, initial_counts: int = 1, **kwargs):
        super().__init__(n_weeks, n_users, n_arms, prices, budget, **kwargs)
        self.explore_rounds = int(explore_rounds)
        
        # Optimistic Initialization
        self.counts = np.full((self.n_users, self.n_arms), initial_counts, dtype=int)
        self.q_values = np.full((self.n_users, self.n_arms), initial_q_value, dtype=float)

        self.round_num = 0
        self.committed = False 
        self.committed_recs = np.full(self.n_users, -1, dtype=int) # Stores the fixed recommendations after commit

    def recommend(self) -> np.ndarray:
        self.round_num += 1
        recommendations = np.full(self.n_users, -1, dtype=int)

        # Build the global feasible set based on current Q-values (mean for greedy selection)
        item_scores_for_feasible_set = np.mean(self.q_values, axis=0)
        feasible_items = build_feasible_set_generic(self.prices, self.budget, item_scores_for_feasible_set)

        if not feasible_items:
            self._store_recommendations(recommendations)
            return recommendations

        feasible_items_np = np.array(feasible_items)

        if self.round_num <= self.explore_rounds and not self.committed:
            # Explore Phase: Randomly select from feasible items
            if len(feasible_items_np) > 0:
                # Distribute feasible items as widely as possible among users, randomly
                if len(feasible_items_np) >= self.n_users:
                    recommendations = np.random.choice(feasible_items_np, size=self.n_users, replace=False)
                else:
                    recommendations = np.random.choice(feasible_items_np, size=self.n_users, replace=True)
        else:
            # Commit Phase: Select based on best observed Q-values
            if not self.committed:
                # Perform commitment once
                for u in range(self.n_users):
                    chosen_arm = -1
                    best_q_value_in_feasible = -np.inf
                    for arm_idx in feasible_items_np:
                        current_item_q_value = self.q_values[u, arm_idx]
                        if current_item_q_value > best_q_value_in_feasible:
                            best_q_value_in_feasible = current_item_q_value
                            chosen_arm = arm_idx
                    self.committed_recs[u] = chosen_arm
                self.committed = True
            
            # After commitment, always return the committed recommendations
            recommendations = self.committed_recs.copy()
        
        self._store_recommendations(recommendations)
        return recommendations

    def update(self, feedback: np.ndarray): # Changed signature
        # Use self.last_recs from BaseBudgetedBandit
        if np.all(self.last_recs == -1):
            return

        # Only update Q-values during the explore phase
        if not self.committed:
            for user_idx, arm_idx, reward in zip(range(self.n_users), self.last_recs, feedback):
                if arm_idx == -1:
                    continue
                
                self.counts[user_idx, arm_idx] += 1
                self.q_values[user_idx, arm_idx] += (reward - self.q_values[user_idx, arm_idx]) / self.counts[user_idx, arm_idx]

    def _get_current_scores(self) -> np.ndarray:
        """
        Returns the current estimated Q-values for all user-arm pairs.
        """
        return self.q_values