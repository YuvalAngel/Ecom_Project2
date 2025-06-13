from Models.BaseModel import *
import numpy as np


class THCR(BaseBudgetedBandit):
    def __init__(self, n_users: int, n_arms: int, prices: np.ndarray, budget: float, n_weeks: int, # n_weeks must be explicitly here due to init signature
                 smoothing: float = 0.5, explore_rounds: int = 50, max_iters: int = 50, **kwargs):
        super().__init__(n_weeks, n_users, n_arms, prices, budget, **kwargs) # Pass n_weeks to super
        
        self.K = self.prices.size # K is derived from prices.size
        
        self.alpha_beta = float(smoothing) # Renamed to avoid clash with alpha parameter in Beta
        self.explore_rounds = int(explore_rounds)
        self.max_iters = int(max_iters)

        self.successes = np.zeros((self.n_users, self.K), dtype=float)
        self.failures = np.zeros((self.n_users, self.K), dtype=float)
        self.round = 0
        self.current_p_hat = None # Stores the sampled p_hat for _get_current_scores

    def __hill_climb(self, initial_S: list, p_hat: np.ndarray) -> list:
        """
        Local search to improve set S under budget using add, remove, and swap moves.
        Finds the best single move (add, remove, or swap) in each iteration.
        """
        current_S = list(initial_S)
        current_S_set = set(current_S)
        
        current_P_contrib = np.max(p_hat[:, current_S], axis=1) if current_S else np.zeros(self.n_users)
        current_gain = current_P_contrib.sum()
        current_cost = self.prices[current_S].sum() if current_S else 0.0

        iters = 0

        while iters < self.max_iters:
            best_move_found_in_iter = False
            best_gain_delta = 0.0 
            
            candidate_next_S = None
            candidate_next_P_contrib = None
            candidate_next_cost = None
            candidate_next_gain = None

            # --- Evaluate Additions ---
            for k_add in range(self.K):
                if k_add not in current_S_set and current_cost + self.prices[k_add] <= self.budget:
                    temp_P_contrib_add = np.maximum(current_P_contrib, p_hat[:, k_add])
                    temp_gain_add = temp_P_contrib_add.sum()
                    
                    if temp_gain_add > current_gain + best_gain_delta:
                        best_gain_delta = temp_gain_add - current_gain
                        
                        candidate_next_S = current_S + [k_add]
                        candidate_next_P_contrib = temp_P_contrib_add
                        candidate_next_cost = current_cost + self.prices[k_add]
                        candidate_next_gain = temp_gain_add
                        best_move_found_in_iter = True
            
            # --- Evaluate Removals ---
            for k_remove in list(current_S): 
                temp_S_after_remove = [item for item in current_S if item != k_remove]
                
                if not temp_S_after_remove:
                    temp_gain_remove = 0.0
                    temp_P_contrib_remove = np.zeros(self.n_users)
                else:
                    temp_P_contrib_remove = np.max(p_hat[:, temp_S_after_remove], axis=1)
                    temp_gain_remove = temp_P_contrib_remove.sum()
                
                if temp_gain_remove > current_gain + best_gain_delta:
                    best_gain_delta = temp_gain_remove - current_gain
                    
                    candidate_next_S = temp_S_after_remove
                    candidate_next_P_contrib = temp_P_contrib_remove
                    candidate_next_cost = current_cost - self.prices[k_remove]
                    candidate_next_gain = temp_gain_remove
                    best_move_found_in_iter = True

            # --- Evaluate Swaps ---
            for old_item_idx, old_item in enumerate(current_S):
                for new_item in range(self.K):
                    if new_item not in current_S_set:
                        temp_cost_swap = current_cost - self.prices[old_item] + self.prices[new_item]
                        
                        if temp_cost_swap <= self.budget:
                            temp_S_after_swap = [item for item in current_S if item != old_item] + [new_item]
                            
                            if not temp_S_after_swap: 
                                temp_gain_swap = 0.0
                                temp_P_contrib_swap = np.zeros(self.n_users)
                            else:
                                temp_P_contrib_swap = np.max(p_hat[:, temp_S_after_swap], axis=1)
                                temp_gain_swap = temp_P_contrib_swap.sum()
                            
                            if temp_gain_swap > current_gain + best_gain_delta:
                                best_gain_delta = temp_gain_swap - current_gain
                                
                                candidate_next_S = temp_S_after_swap
                                candidate_next_P_contrib = temp_P_contrib_swap
                                candidate_next_cost = temp_cost_swap
                                candidate_next_gain = temp_gain_swap
                                best_move_found_in_iter = True

            if best_move_found_in_iter:
                current_S = candidate_next_S
                current_S_set = set(current_S)
                current_P_contrib = candidate_next_P_contrib
                current_gain = candidate_next_gain
                current_cost = candidate_next_cost
                iters += 1
            else:
                break 

        return current_S

    def recommend(self) -> np.ndarray:
        self.round += 1
        
        if self.round <= self.explore_rounds:
            random_scores = np.random.rand(self.n_users, self.K)
            S = build_feasible_set_generic(self.prices, self.budget, random_scores)
            self.current_p_hat = random_scores # Store these scores for _get_current_scores during exploration

            if not S:
                recs = np.full(self.n_users, -1)
            else:
                if len(S) >= self.n_users:
                    recs = np.random.choice(S, size=self.n_users, replace=False)
                else:
                    recs = np.random.choice(S, size=self.n_users, replace=True)
            
            self._store_recommendations(recs)
            return recs

        # After exploration, sample p_hat
        p_hat = np.random.beta(self.successes + self.alpha_beta, self.failures + self.alpha_beta)
        self.current_p_hat = p_hat # Store the actual sampled p_hat for _get_current_scores
        
        initial_S = build_feasible_set_generic(self.prices, self.budget, p_hat)
        best_S = self.__hill_climb(initial_S, p_hat)

        if not best_S:
            recs = np.full(self.n_users, -1)
        else:
            p_hat_S = p_hat[:, best_S]
            best_indices_in_S = np.argmax(p_hat_S, axis=1)
            recs = np.array([best_S[i] for i in best_indices_in_S])
        
        self._store_recommendations(recs)
        return recs

    def update(self, feedback: np.ndarray):
        valid_mask = (self.last_recs != -1)
        users_for_update = np.arange(self.n_users)[valid_mask]
        arms_for_update = self.last_recs[valid_mask]
        feedback_for_update = feedback[valid_mask]

        success_mask = (feedback_for_update == 1)
        failure_mask = (feedback_for_update == 0)

        self.successes[users_for_update[success_mask], arms_for_update[success_mask]] += 1.0
        self.failures[users_for_update[failure_mask], arms_for_update[failure_mask]] += 1.0
    
    def _get_current_scores(self) -> np.ndarray:
        """
        Returns the sampled p_hat values for the current round, 
        or a proxy if not yet sampled (e.g., initial state or exploration phase).
        """
        if self.current_p_hat is not None:
            return self.current_p_hat
        else:
            # Fallback for very first round before recommend() is called
            return (self.successes + self.alpha_beta) / (self.successes + self.failures + 2 * self.alpha_beta + 1e-8)


class UCB_MB(BaseBudgetedBandit):
    def __init__(self, n_weeks: int, n_users: int, n_arms: int, prices: np.ndarray, budget: float, c: float = 0.1, initial_q_value: float = 1.0, initial_counts: int = 1, **kwargs):
        super().__init__(n_weeks, n_users, n_arms, prices, budget, **kwargs)
        self.c = c # Exploration parameter
        self.t = 0  # Global time step (number of rounds played)

        # User-specific counts and rewards (for mean reward calculation)
        self.arm_counts = np.full((n_users, n_arms), initial_counts, dtype=float)
        # Store sum of rewards for each user-arm pair
        self.arm_rewards = np.full((n_users, n_arms), initial_q_value * initial_counts, dtype=float)
        
        # User-specific Q-values (mean rewards)
        self.q_values = np.full((n_users, n_arms), initial_q_value, dtype=float)

    def _calculate_ucb_scores(self) -> np.ndarray:
        """Helper to calculate UCB values based on current state."""
        ucb_scores = np.zeros((self.n_users, self.n_arms))
        log_t = np.log(self.t) if self.t > 0 else 0.0 # Handle t=0

        for u in range(self.n_users):
            for arm_idx in range(self.n_arms):
                n_ua = self.arm_counts[u, arm_idx]
                q_ua = self.q_values[u, arm_idx]

                if n_ua == 0:
                    ucb_scores[u, arm_idx] = float('inf')
                else:
                    empirical_variance = q_ua * (1 - q_ua)
                    if empirical_variance < 1e-9:
                        empirical_variance = 0.25
                    
                    bound = self.c * np.sqrt(2 * empirical_variance * log_t / n_ua)
                    ucb_scores[u, arm_idx] = q_ua + bound
        return ucb_scores

    def recommend(self) -> np.ndarray:
        self.t += 1
        recommended_arms_for_users = np.full(self.n_users, -1, dtype=int)

        ucb_values_for_round = self._calculate_ucb_scores() 

        feasible_set_S = build_feasible_set_generic(self.prices, self.budget, ucb_values_for_round, agg_fn=np.mean)

        if not feasible_set_S:
            self._store_recommendations(recommended_arms_for_users)
            return recommended_arms_for_users
            
        for u in range(self.n_users):
            if feasible_set_S:
                chosen_arm = max(feasible_set_S, key=lambda a: ucb_values_for_round[u, a])
                recommended_arms_for_users[u] = chosen_arm
        
        self._store_recommendations(recommended_arms_for_users)
        return recommended_arms_for_users

    def update(self, results: np.array):
        # We need to make sure _last_recommended_arms is stored via super()._store_recommendations()
        # This check should ideally be done in _store_recommendations if needed, or assume it's always set.
        if not hasattr(self, 'last_recs') or np.all(self.last_recs == -1): # Check if any valid recs exist
             return

        for user_idx, arm_idx in enumerate(self.last_recs): # Use self.last_recs
            if arm_idx == -1:
                continue
            reward = results[user_idx]
            self.arm_counts[user_idx, arm_idx] += 1
            self.arm_rewards[user_idx, arm_idx] += reward
            self.q_values[user_idx, arm_idx] = self.arm_rewards[user_idx, arm_idx] / self.arm_counts[user_idx, arm_idx]
    
    def _get_current_scores(self) -> np.ndarray:
        return self._calculate_ucb_scores()
