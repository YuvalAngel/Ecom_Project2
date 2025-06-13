from Models.BaseModel import *
import numpy as np

class LinUCB(BaseBudgetedBandit):
    def __init__(self, n_weeks: int, n_users: int, n_arms: int, prices: np.ndarray, budget: float,
                 item_features: np.ndarray = None, alpha: float = 0.5, lambda_reg: float = 1.0,
                 default_feature_dim: int = 10, default_feature_scale: float = 0.1, **kwargs):
        super().__init__(n_weeks, n_users, n_arms, prices, budget, **kwargs)
        self.alpha = float(alpha)
        self.lambda_reg = float(lambda_reg)

        # Handle item_features: use provided, or generate default random ones
        if item_features is None:
            self.item_features = np.random.rand(self.n_arms, default_feature_dim) * default_feature_scale
            self.feature_dim = default_feature_dim
        else:
            self.item_features = item_features
            self.feature_dim = item_features.shape[1]

        # A_u: (D, D) matrix for each user, accumulates x_i * x_i^T
        # Initialize with lambda_reg * I for ridge regression
        self.A = np.array([lambda_reg * np.eye(self.feature_dim) for _ in range(n_users)])
        # b_u: (D,) vector for each user, accumulates r_i * x_i
        self.b = np.array([np.zeros(self.feature_dim) for _ in range(n_users)])
        
        # Store last computed UCB scores for _get_current_scores
        self._last_ucb_scores = np.zeros((self.n_users, self.n_arms))

    def _calculate_linucb_scores(self) -> np.ndarray:
        """Helper to calculate LinUCB scores for all user-arm pairs."""
        ucb_scores = np.zeros((self.n_users, self.n_arms))
        
        for u in range(self.n_users):
            try:
                A_u_inv = np.linalg.inv(self.A[u])
            except np.linalg.LinAlgError:
                # Handle singular matrix case, e.g., by returning 0 or a very high value for unplayed arms
                A_u_inv = np.linalg.pinv(self.A[u]) # Use pseudo-inverse as fallback
                
            theta_u = A_u_inv @ self.b[u] 

            for arm_idx in range(self.n_arms):
                x_a = self.item_features[arm_idx] 
                mean_reward = theta_u @ x_a
                variance_term = x_a @ A_u_inv @ x_a
                ucb_score = mean_reward + self.alpha * np.sqrt(variance_term)
                ucb_scores[u, arm_idx] = ucb_score
        return ucb_scores

    def recommend(self) -> np.ndarray:
        recommendations = np.full(self.n_users, -1, dtype=int)
        
        self._last_ucb_scores = self._calculate_linucb_scores() # Calculate and store UCB scores

        item_scores_for_feasible_set = np.mean(self._last_ucb_scores, axis=0)
        feasible_items = build_feasible_set_generic(self.prices, self.budget, item_scores_for_feasible_set)

        if not feasible_items:
            self._store_recommendations(recommendations)
            return recommendations

        feasible_items_np = np.array(feasible_items)

        for u in range(self.n_users):
            chosen_arm = -1
            best_ucb_value = -np.inf
            
            for arm_idx in feasible_items_np:
                current_ucb_value = self._last_ucb_scores[u, arm_idx]
                if current_ucb_value > best_ucb_value:
                    best_ucb_value = current_ucb_value
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
            
            x_a = self.item_features[arm_idx]
            self.A[user_idx] += np.outer(x_a, x_a)
            self.b[user_idx] += reward * x_a

    def _get_current_scores(self) -> np.ndarray:
        """
        Returns the last computed LinUCB scores for all user-arm pairs.
        """
        # If recommend() hasn't been called yet, calculate them.
        if self._last_ucb_scores is None or np.all(self._last_ucb_scores == 0):
             return self._calculate_linucb_scores()
        return self._last_ucb_scores