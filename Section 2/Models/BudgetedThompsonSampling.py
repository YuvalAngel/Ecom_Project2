from Models.BaseModel import *
import numpy as np

class BudgetedThompsonSampling(BaseBudgetedBandit):
    def __init__(self, n_weeks: int, n_users: int, n_arms: int, prices: np.ndarray, budget: float, alpha_prior: float = 1.0, beta_prior: float = 1.0, **kwargs):
        super().__init__(n_weeks, n_users, n_arms, prices, budget, **kwargs)
        
        # User-specific alpha and beta parameters for Beta distribution (Bernoulli conjugate prior)
        self.alpha = np.full((n_users, n_arms), alpha_prior, dtype=float) # Number of successes + alpha_prior
        self.beta = np.full((n_users, n_arms), beta_prior, dtype=float)   # Number of failures + beta_prior

    def recommend(self) -> np.ndarray:
        recommended_arms_for_users = np.full(self.n_users, -1, dtype=int)
        
        sampled_probabilities_per_user = np.zeros((self.n_users, self.n_arms))
        for u in range(self.n_users):
            for arm_idx in range(self.n_arms):
                sampled_probabilities_per_user[u, arm_idx] = np.random.beta(self.alpha[u, arm_idx], self.beta[u, arm_idx])

        feasible_set_S = build_feasible_set_generic(self.prices, self.budget, sampled_probabilities_per_user, agg_fn=np.mean)

        if not feasible_set_S:
            self._store_recommendations(recommended_arms_for_users)
            return recommended_arms_for_users

        for u in range(self.n_users):
            if feasible_set_S:
                chosen_arm = max(feasible_set_S, key=lambda a: sampled_probabilities_per_user[u, a])
                recommended_arms_for_users[u] = chosen_arm
            else:
                pass 
        
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
            self.alpha[user_idx, arm_idx] += reward 
            self.beta[user_idx, arm_idx] += (1 - reward) 

    def _get_current_scores(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta + 1e-8)
