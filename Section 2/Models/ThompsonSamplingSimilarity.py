from Models.BaseModel import *
import numpy as np

class ThompsonSamplingSimilarity(BaseBudgetedBandit):
    def __init__(self, n_weeks: int, n_users: int, n_arms: int, prices: np.ndarray, budget: float, 
                 alpha_prior: float = 1.0, beta_prior: float = 1.0, 
                 similarity_matrix: np.ndarray = None, 
                 default_sim_type: str = 'block', sim_group_size: int = 5, 
                 sim_self_weight: float = 0.7, sim_cross_weight: float = 0.05, **kwargs):
        super().__init__(n_weeks, n_users, n_arms, prices, budget, **kwargs)
        self.successes = np.zeros((self.n_users, self.n_arms), dtype=float)
        self.failures = np.zeros((self.n_users, self.n_arms), dtype=float)
        self.alpha_prior = alpha_prior 
        self.beta_prior = beta_prior 
        self._last_smoothed_samples = None # Store for _get_current_scores

        # Generate or normalize similarity matrix
        if similarity_matrix is None:
            if default_sim_type == 'identity':
                self.similarity = np.eye(n_users) 
            elif default_sim_type == 'uniform':
                self.similarity = np.full((n_users, n_users), 1.0 / n_users)
            elif default_sim_type == 'block':
                self.similarity = generate_block_similarity(n_users, sim_group_size, sim_self_weight, sim_cross_weight)
            else:
                raise ValueError(f"Unknown default_sim_type: '{default_sim_type}'. Choose from 'identity', 'uniform', 'block'.")
        else:
            # Normalize the explicitly provided matrix to ensure rows sum to 1 for weighted average
            row_sums = similarity_matrix.sum(axis=1, keepdims=True)
            self.similarity = similarity_matrix / (row_sums + 1e-8) 

    def recommend(self) -> np.ndarray:
        samples_raw = np.random.beta(self.successes + self.alpha_prior, self.failures + self.beta_prior)
        samples_smoothed = self.similarity @ samples_raw
        self._last_smoothed_samples = samples_smoothed # Store for _get_current_scores
        
        feasible_set_S = build_feasible_set_generic(self.prices, self.budget, samples_smoothed, agg_fn=np.mean)

        recommendations = np.full(self.n_users, -1, dtype=int)
        if not feasible_set_S:
            self._store_recommendations(recommendations)
            return recommendations
        else:
            for u in range(self.n_users):
                chosen_arm = max(feasible_set_S, key=lambda a: samples_smoothed[u, a])
                recommendations[u] = chosen_arm
        
        self._store_recommendations(recommendations)
        return recommendations

    def update(self, feedback: np.ndarray): # Changed signature
        # Use self.last_recs from BaseBudgetedBandit
        if np.all(self.last_recs == -1):
            return

        for user_idx, arm_idx, reward in zip(range(self.n_users), self.last_recs, feedback):
            if arm_idx == -1:
                continue
            
            if reward > 0: # Reward is 1 (success)
                self.successes[user_idx, arm_idx] += 1
            else: # Reward is 0 (failure)
                self.failures[user_idx, arm_idx] += 1
    
    def _get_current_scores(self) -> np.ndarray:
        """
        Returns the last smoothed sampled probabilities. If no recommendations made yet,
        returns the mean of the current Beta distributions (after smoothing).
        """
        if self._last_smoothed_samples is not None:
            return self._last_smoothed_samples
        else:
            # Fallback for very first round or if no samples yet
            mean_beta = (self.successes + self.alpha_prior) / (self.successes + self.failures + self.alpha_prior + self.beta_prior + 1e-8)
            return self.similarity @ mean_beta