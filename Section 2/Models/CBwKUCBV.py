from Models.BaseModel import *
import numpy as np

class CBwKUCBV(BaseBudgetedBandit):
    """
    Budgeted Bandit using UCB-V, which incorporates empirical variance into the confidence bound,
    combined with a greedy knapsack solver.
    Assumes rewards are in [0, 1].
    """
    def __init__(self, n_weeks: int, n_users: int, n_arms: int, prices: np.ndarray, budget: float,
                 alpha: float = 0.1,  # Parameter for UCB-V (often set small, e.g., 0.1)
                 epsilon: float = 1e-6, # Small value to avoid division by zero/log(0)
                 add_noise_to_knapsack: bool = True, knapsack_noise_scale: float = 1e-4, **kwargs):
        super().__init__(n_weeks, n_users, n_arms, prices, budget, **kwargs)
        self.alpha = alpha
        self.epsilon = epsilon # For numerical stability in log terms

        self.n_plays = np.zeros((self.n_users, self.n_arms), dtype=int)
        self.s_rewards = np.zeros((self.n_users, self.n_arms))
        self.s_rewards_sq = np.zeros((self.n_users, self.n_arms)) # Sum of squared rewards for variance

        self.add_noise_to_knapsack = add_noise_to_knapsack
        self.knapsack_noise_scale = knapsack_noise_scale

    def _get_current_scores(self) -> np.ndarray:
        ucb_scores = np.zeros((self.n_users, self.n_arms))

        n_plays_safe = np.maximum(self.n_plays, 1) # Avoid division by zero
        mu_hat = self.s_rewards / n_plays_safe
        
        # Empirical variance: var_hat = (sum_sq_rewards / n_plays) - mu_hat^2
        # Clamp variance to be non-negative due to floating point inaccuracies
        var_hat = np.maximum(0, (self.s_rewards_sq / n_plays_safe) - (mu_hat ** 2))

        # UCB-V confidence bound: sqrt(2 * var_hat * log(t) / n_plays) + 3 * log(t) / n_plays
        # Assumes rewards are in [0, 1]. If not, constants might need adjustment.
        log_t = np.log(np.maximum(1, self.current_week) + self.epsilon) # t is current_week (or round)

        # C_t is the empirical variance part
        c_t_part = np.sqrt(2 * var_hat * log_t / n_plays_safe)
        
        # d_t is the small bias term for bounded rewards
        d_t_part = 3 * log_t / n_plays_safe

        confidence = c_t_part + d_t_part

        ucb_scores = mu_hat + self.alpha * confidence # Alpha scales the entire confidence term

        # For arms not played yet, give an optimistic initial UCB (e.g., 1.0)
        ucb_scores[self.n_plays == 0] = 1.0 
        
        return ucb_scores

    def recommend(self) -> np.ndarray:
        ucb_scores = self._get_current_scores()

        purchased_arms_this_week = np.array(
            build_feasible_set_generic(
                prices=self.prices,
                budget=self.budget,
                scores=ucb_scores,
                agg_fn=np.mean, # Not directly used by current build_feasible_set_generic
                add_noise=self.add_noise_to_knapsack,
                noise_scale=self.knapsack_noise_scale
            )
        )
        
        if len(purchased_arms_this_week) == 0:
            recommendations = np.full(self.n_users, -1, dtype=int)
        else:
            recommendations = np.full(self.n_users, -1, dtype=int)
            for user_idx in range(self.n_users):
                user_ucb_for_purchased_arms = ucb_scores[user_idx, purchased_arms_this_week]
                
                if user_ucb_for_purchased_arms.size > 0:
                    best_arm_idx_in_purchased = np.argmax(user_ucb_for_purchased_arms)
                    recommendations[user_idx] = purchased_arms_this_week[best_arm_idx_in_purchased]

        self._store_recommendations(recommendations)
        return recommendations

    def update(self, feedback: np.ndarray):
        self.current_week += 1

        for user_idx in range(self.n_users):
            recommended_arm = self.last_recs[user_idx]
            reward = feedback[user_idx]

            if recommended_arm != -1: 
                self.n_plays[user_idx, recommended_arm] += 1
                self.s_rewards[user_idx, recommended_arm] += reward
                self.s_rewards_sq[user_idx, recommended_arm] += reward**2

