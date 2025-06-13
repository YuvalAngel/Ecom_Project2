from Models.BaseModel import *
import numpy as np

class MF_UCB(BaseBudgetedBandit):
    def __init__(self, n_weeks: int, n_users: int, n_arms: int, prices: np.ndarray, budget: float,
                 latent_dim: int = 10, learning_rate: float = 0.05, alpha: float = 0.5, lambda_reg: float = 0.05, **kwargs):
        super().__init__(n_weeks, n_users, n_arms, prices, budget, **kwargs)
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.alpha = float(alpha)
        self.lambda_reg = float(lambda_reg)

        # Initialize user and item latent factors randomly
        self.user_embeddings = np.random.rand(n_users, latent_dim) * 0.1
        self.item_embeddings = np.random.rand(n_arms, latent_dim) * 0.1
        
        # Counts for UCB bonus (optimistic initialization)
        self.counts = np.full((self.n_users, self.n_arms), 1.0) 
        
        self.round_num = 0
        self._last_mf_ucb_scores = None # Store for _get_current_scores

    def _predict_reward(self, user_idx: int, arm_idx: int) -> float:
        """Helper to predict reward for a single user-arm pair."""
        return np.dot(self.user_embeddings[user_idx], self.item_embeddings[arm_idx])

    def _calculate_mf_ucb_scores(self) -> np.ndarray:
        """Helper to calculate MF-UCB scores for all user-arm pairs."""
        mf_ucb_scores = np.zeros((self.n_users, self.n_arms))
        
        for u in range(self.n_users):
            for arm_idx in range(self.n_arms):
                predicted_mean = self._predict_reward(u, arm_idx)
                
                # Clip predicted mean to [0, 1] for stability with binary rewards
                predicted_mean = np.clip(predicted_mean, 0.0, 1.0)

                # UCB-like bonus: alpha * sqrt(log(round_num) / N_ua)
                exploration_bonus = self.alpha * np.sqrt(np.log(self.round_num + 1e-8) / (self.counts[u, arm_idx] + 1e-8))
                
                mf_ucb_scores[u, arm_idx] = predicted_mean + exploration_bonus
        return mf_ucb_scores

    def recommend(self) -> np.ndarray:
        self.round_num += 1
        recommendations = np.full(self.n_users, -1, dtype=int)

        self._last_mf_ucb_scores = self._calculate_mf_ucb_scores() # Calculate and store scores
        
        # Build the global feasible set using the mean MF-UCB score for each item across users
        item_scores_for_feasible_set = np.mean(self._last_mf_ucb_scores, axis=0)
        feasible_items = build_feasible_set_generic(self.prices, self.budget, item_scores_for_feasible_set)

        if not feasible_items:
            self._store_recommendations(recommendations)
            return recommendations

        feasible_items_np = np.array(feasible_items)

        for u in range(self.n_users):
            chosen_arm = -1
            best_score = -np.inf
            
            # Select the arm from the feasible set with the highest MF-UCB score for this user
            for arm_idx in feasible_items_np:
                current_score = self._last_mf_ucb_scores[u, arm_idx]
                if current_score > best_score:
                    best_score = current_score
                    chosen_arm = arm_idx
            
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
            
            self.counts[user_idx, arm_idx] += 1
            
            predicted_reward = self._predict_reward(user_idx, arm_idx)
            error = reward - predicted_reward
            
            # Update embeddings using SGD
            self.user_embeddings[user_idx] += self.learning_rate * error * self.item_embeddings[arm_idx]
            self.item_embeddings[arm_idx] += self.learning_rate * error * self.user_embeddings[user_idx]
            
            # Add L2 regularization to prevent overfitting
            self.user_embeddings[user_idx] -= self.learning_rate * self.lambda_reg * self.user_embeddings[user_idx]
            self.item_embeddings[arm_idx] -= self.learning_rate * self.lambda_reg * self.item_embeddings[arm_idx]

    def _get_current_scores(self) -> np.ndarray:
        """
        Returns the last computed MF-UCB scores for all user-arm pairs.
        """
        if self._last_mf_ucb_scores is None or np.all(self._last_mf_ucb_scores == 0):
            return self._calculate_mf_ucb_scores()
        return self._last_mf_ucb_scores