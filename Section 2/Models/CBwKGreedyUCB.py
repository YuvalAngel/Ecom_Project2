from Models.BaseModel import *
import numpy as np

class CBwKGreedyUCB(BaseBudgetedBandit):
    def __init__(self, n_weeks: int, n_users: int, n_arms: int, prices: np.ndarray, budget: float, alpha: float = 0.5, **kwargs):
        super().__init__(n_weeks, n_users, n_arms, prices, budget, **kwargs)
        self.alpha = alpha

        self.n_plays = np.zeros((self.n_users, self.n_arms), dtype=int)
        self.s_rewards = np.zeros((self.n_users, self.n_arms))

        self.current_week = 0
        self.purchased_arms_this_week = np.array([], dtype=int)

    def _get_current_scores(self) -> np.ndarray:
        n_plays_safe = np.maximum(self.n_plays, 1) 
        mu_hat = self.s_rewards / n_plays_safe
        
        # --- IMPORTANT CHANGE FROM PREVIOUS VERSION ---
        # Using current_week + 1 for the 't' in UCB, as discussed, to control exploration.
        total_rounds = self.current_week + 1 
        
        confidence = self.alpha * np.sqrt(2 * np.log(total_rounds) / n_plays_safe)
        
        ucb_scores = mu_hat + confidence
        
        # For arms not played yet, give an optimistic initial UCB (e.g., 1.0)
        ucb_scores[self.n_plays == 0] = 1.0 
        
        return ucb_scores

    def recommend(self) -> np.ndarray:
        ucb_scores = self._get_current_scores()

        # --- IMPORTANT CHANGE IN CALLING build_feasible_set_generic ---
        # The new build_feasible_set_generic implicitly handles aggregation
        # and doesn't use agg_fn or add_noise in its new logic,
        # but we pass them for signature compatibility.
        self.purchased_arms_this_week = np.array(
            build_feasible_set_generic(
                prices=self.prices,
                budget=self.budget,
                scores=ucb_scores, # Pass the full UCB score matrix
                agg_fn=np.mean, # Kept for signature compatibility
                add_noise=False # Kept for signature compatibility
            )
        )
        
        if len(self.purchased_arms_this_week) == 0:
            recommendations = np.full(self.n_users, -1, dtype=int)
        else:
            recommendations = np.full(self.n_users, -1, dtype=int)
            for user_idx in range(self.n_users):
                user_ucb_for_purchased_arms = ucb_scores[user_idx, self.purchased_arms_this_week]
                
                if user_ucb_for_purchased_arms.size > 0:
                    best_arm_idx_in_purchased = np.argmax(user_ucb_for_purchased_arms)
                    recommendations[user_idx] = self.purchased_arms_this_week[best_arm_idx_in_purchased]

        self._store_recommendations(recommendations)
        return recommendations

    def update(self, feedback: np.ndarray):
        self.current_week += 1

        for user_idx in range(self.n_users):
            recommended_arm = self.last_recs[user_idx]
            if recommended_arm != -1: 
                self.n_plays[user_idx, recommended_arm] += 1
                self.s_rewards[user_idx, recommended_arm] += feedback[user_idx]