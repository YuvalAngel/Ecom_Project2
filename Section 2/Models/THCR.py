from Models.BaseModel import *
import numpy as np


class THCR(BaseBudgetedBandit):
    """
    Thompson Hill Climb Recommender (THCR) for the Budgeted Multi-Armed Bandit problem.

    This model combines Thompson Sampling for estimating user-arm enjoyment probabilities
    with a greedy hill-climbing local search algorithm to select a feasible set of arms
    that maximizes the expected total enjoyment, given a per-round budget.
    It includes an initial pure exploration phase.
    Improvements: Multiple random restarts for hill climbing and a random walk component
    to escape local optima, prioritizing cumulative reward over efficiency.
    Optimized for small N_users and N_arms by leveraging simple, fast NumPy operations.
    """
    def __init__(self, n_weeks: int, n_users: int, n_arms: int, prices: np.ndarray, budget: float,
                 smoothing: float = 0.5, explore_rounds: int = 50, max_iters: int = 50,
                 n_hill_climb_restarts: int = 5, random_walk_prob: float = 0.05,
                 initial_set_noise_scale: float = 0.01,
                 **kwargs):
        """
        Initializes the THCR model.

        Args:
            n_weeks (int): Total number of weeks (rounds) for the simulation.
            n_users (int): Number of users.
            n_arms (int): Number of available arms (items).
            prices (np.ndarray): A 1D NumPy array of costs for each arm (shape: n_arms,).
            budget (float): The total budget available for purchasing arms in each round.
            smoothing (float): The alpha and beta parameters for the Beta-Bernoulli prior
                               (Beta(alpha+1, beta+1)). Common values are 0.5 for Jeffreys prior,
                               or 1.0 for uniform prior. Must be non-negative.
            explore_rounds (int): The number of initial rounds during which the model
                                  performs pure random exploration. Must be non-negative.
            max_iters (int): The maximum number of iterations for the hill-climbing local search.
                             Must be a positive integer.
            n_hill_climb_restarts (int): Number of times to run the hill-climbing algorithm
                                         from different initial sets. The best result among these
                                         restarts will be chosen. Must be a positive integer.
            random_walk_prob (float): Probability (between 0 and 1) of accepting a non-improving
                                      move during hill climbing to escape local optima.
            initial_set_noise_scale (float): Standard deviation of Gaussian noise added to scores
                                             when building initial sets for hill climbing restarts.
            **kwargs: Additional keyword arguments passed to the BaseModel.
        
        Raises:
            ValueError: If smoothing is negative, explore_rounds is negative, max_iters is not positive,
                        n_hill_climb_restarts is not positive, or random_walk_prob is not between 0 and 1.
        """
        super().__init__(n_weeks, n_users, n_arms, prices, budget, **kwargs)
        
        if smoothing < 0:
            raise ValueError("Smoothing parameter (alpha_beta) must be non-negative.")
        if explore_rounds < 0:
            raise ValueError("Explore rounds must be non-negative.")
        if not (isinstance(max_iters, int) and max_iters > 0):
            raise ValueError("Max iterations for hill climbing must be a positive integer.")
        if not (isinstance(n_hill_climb_restarts, int) and n_hill_climb_restarts > 0):
            raise ValueError("Number of hill climb restarts must be a positive integer.")
        if not (0.0 <= random_walk_prob <= 1.0):
            raise ValueError("Random walk probability must be between 0 and 1.")

        self.K = self.n_arms
        
        self.alpha_beta = smoothing 
        self.explore_rounds = explore_rounds
        self.max_iters = max_iters
        self.n_hill_climb_restarts = n_hill_climb_restarts
        self.random_walk_prob = random_walk_prob
        self.initial_set_noise_scale = initial_set_noise_scale

        self.successes = np.zeros((self.n_users, self.K), dtype=float)
        self.failures = np.zeros((self.n_users, self.K), dtype=float)
        
        self.round = 0
        self.current_p_hat = None

    def __hill_climb(self, initial_S: list, p_hat: np.ndarray) -> list:
        """
        Performs a local search (hill climbing) to improve a given set of arms 'S'
        under the budget constraint, maximizing the sum of expected enjoyment across users.
        The search considers single add, remove, and swap moves. Includes a random walk
        component to escape local optima. This version is optimized for smaller N/K
        by relying on simpler, efficient NumPy array operations for full slice calculations.

        Args:
            initial_S (list): The starting list of arm indices for the local search.
            p_hat (np.ndarray): The sampled (n_users, n_arms) probability matrix
                                 representing estimated enjoyment for each user-arm pair.

        Returns:
            list: The optimized list of arm indices (S) after the hill-climbing process.
        """
        current_S = list(initial_S)
        current_S_set = set(current_S) # Maintain set for faster lookups
        
        current_P_contrib = np.max(p_hat[:, current_S], axis=1) if current_S else np.zeros(self.n_users)
        current_gain = current_P_contrib.sum()
        current_cost = self.prices[current_S].sum() if current_S else 0.0

        iters = 0

        while iters < self.max_iters:
            best_strict_improvement_found_in_iter = False
            best_gain_delta_strict = 0.0 
            
            candidate_next_S_overall = None
            candidate_next_P_contrib_overall = None
            candidate_next_cost_overall = None
            candidate_next_gain_overall = -np.inf # Initialize with a very low value for comparison

            # --- Evaluate Additions ---
            for k_add in range(self.K):
                if k_add not in current_S_set:
                    cost_if_added = current_cost + self.prices[k_add]
                    if cost_if_added <= self.budget:
                        temp_P_contrib_add = np.maximum(current_P_contrib, p_hat[:, k_add])
                        temp_gain_add = temp_P_contrib_add.sum()
                        
                        # Check for strict improvement
                        if temp_gain_add > current_gain + best_gain_delta_strict:
                            best_gain_delta_strict = temp_gain_add - current_gain
                            best_strict_improvement_found_in_iter = True
                            
                            candidate_next_S_overall = current_S + [k_add]
                            candidate_next_P_contrib_overall = temp_P_contrib_add
                            candidate_next_cost_overall = cost_if_added
                            candidate_next_gain_overall = temp_gain_add
                        
                        # Check if this is the best non-worsening move found so far
                        elif temp_gain_add > candidate_next_gain_overall: 
                            candidate_next_S_overall = current_S + [k_add]
                            candidate_next_P_contrib_overall = temp_P_contrib_add
                            candidate_next_cost_overall = cost_if_added
                            candidate_next_gain_overall = temp_gain_add
            
            # --- Evaluate Removals ---
            for k_remove in list(current_S): # Iterate over a copy to modify current_S
                temp_S_after_remove = [item for item in current_S if item != k_remove]
                
                if not temp_S_after_remove:
                    temp_gain_remove = 0.0
                    temp_P_contrib_remove = np.zeros(self.n_users)
                else:
                    # Rely on numpy's optimized max for the (small) slice
                    temp_P_contrib_remove = np.max(p_hat[:, temp_S_after_remove], axis=1)
                    temp_gain_remove = temp_P_contrib_remove.sum()
                
                # Check for strict improvement
                if temp_gain_remove > current_gain + best_gain_delta_strict:
                    best_gain_delta_strict = temp_gain_remove - current_gain
                    best_strict_improvement_found_in_iter = True
                    
                    candidate_next_S_overall = temp_S_after_remove
                    candidate_next_P_contrib_overall = temp_P_contrib_remove
                    candidate_next_cost_overall = current_cost - self.prices[k_remove]
                    candidate_next_gain_overall = temp_gain_remove

                # Check if this is the best non-worsening move found so far
                elif temp_gain_remove > candidate_next_gain_overall:
                    candidate_next_S_overall = temp_S_after_remove
                    candidate_next_P_contrib_overall = temp_P_contrib_remove
                    candidate_next_cost_overall = current_cost - self.prices[k_remove]
                    candidate_next_gain_overall = temp_gain_remove

            # --- Evaluate Swaps ---
            for old_item in current_S:
                for new_item in range(self.K):
                    if new_item not in current_S_set:
                        temp_cost_swap = current_cost - self.prices[old_item] + self.prices[new_item]
                        
                        if temp_cost_swap <= self.budget:
                            temp_S_after_swap = [item for item in current_S if item != old_item] + [new_item]
                            
                            if not temp_S_after_swap: 
                                temp_gain_swap = 0.0
                                temp_P_contrib_swap = np.zeros(self.n_users)
                            else:
                                # Rely on numpy's optimized max for the (small) slice
                                temp_P_contrib_swap = np.max(p_hat[:, temp_S_after_swap], axis=1)
                                temp_gain_swap = temp_P_contrib_swap.sum()
                            
                            # Check for strict improvement
                            if temp_gain_swap > current_gain + best_gain_delta_strict:
                                best_gain_delta_strict = temp_gain_swap - current_gain
                                best_strict_improvement_found_in_iter = True
                                
                                candidate_next_S_overall = temp_S_after_swap
                                candidate_next_P_contrib_overall = temp_P_contrib_swap
                                candidate_next_cost_overall = temp_cost_swap
                                candidate_next_gain_overall = temp_gain_swap
                            
                            # Check if this is the best non-worsening move found so far
                            elif temp_gain_swap > candidate_next_gain_overall:
                                candidate_next_S_overall = temp_S_after_swap
                                candidate_next_P_contrib_overall = temp_P_contrib_swap
                                candidate_next_cost_overall = temp_cost_swap
                                candidate_next_gain_overall = temp_gain_swap

            # --- Apply best move or random walk ---
            if best_strict_improvement_found_in_iter:
                # Always take a strictly improving move
                current_S = candidate_next_S_overall
                current_S_set = set(current_S)
                current_P_contrib = candidate_next_P_contrib_overall
                current_gain = candidate_next_gain_overall
                current_cost = candidate_next_cost_overall
                iters += 1
            elif candidate_next_S_overall is not None and np.random.rand() < self.random_walk_prob:
                # No strict improvement, but a non-worse move found, accept with random_walk_prob
                current_S = candidate_next_S_overall
                current_S_set = set(current_S)
                current_P_contrib = candidate_next_P_contrib_overall
                current_gain = candidate_next_gain_overall
                current_cost = candidate_next_cost_overall
                iters += 1
            else:
                # No improving move found (or random walk not taken), stop
                break 

        return current_S

    # ... (recommend, update, _get_current_scores remain the same as the previous version with restarts)
    def recommend(self) -> np.ndarray:
        self.round += 1
        recommended_arms_for_users = np.full(self.n_users, -1, dtype=int)
        
        if self.round <= self.explore_rounds:
            random_scores = np.random.rand(self.n_users, self.K) 
            S = build_feasible_set_generic(self.prices, self.budget, random_scores, add_noise=True, noise_scale=0.1)
            
            self.current_p_hat = random_scores

            if S: 
                recommended_arms_for_users = np.random.choice(S, size=self.n_users, replace=True)
            
            self._store_recommendations(recommended_arms_for_users)
            return recommended_arms_for_users

        p_hat = np.random.beta(self.successes + self.alpha_beta, self.failures + self.alpha_beta)
        self.current_p_hat = p_hat

        overall_best_S = []
        overall_max_gain = -np.inf

        for _ in range(self.n_hill_climb_restarts):
            restart_initial_S = build_feasible_set_generic(
                self.prices, self.budget, p_hat, agg_fn=np.mean, 
                add_noise=True, noise_scale=self.initial_set_noise_scale
            )
            
            current_restart_S = self.__hill_climb(restart_initial_S, p_hat)

            if current_restart_S:
                current_restart_P_contrib = np.max(p_hat[:, current_restart_S], axis=1)
                current_restart_gain = current_restart_P_contrib.sum()
            else:
                current_restart_gain = 0.0

            if current_restart_gain > overall_max_gain:
                overall_max_gain = current_restart_gain
                overall_best_S = current_restart_S

        if not overall_best_S:
            pass 
        else:
            p_hat_S = p_hat[:, overall_best_S]
            best_indices_in_S = np.argmax(p_hat_S, axis=1)
            recommended_arms_for_users = np.array([overall_best_S[i] for i in best_indices_in_S])
        
        self._store_recommendations(recommended_arms_for_users)
        return recommended_arms_for_users

    def update(self, feedback: np.ndarray):
        if not (isinstance(feedback, np.ndarray) and feedback.shape == (self.n_users,)):
            raise ValueError(f"Feedback must be a 1D numpy array of shape ({self.n_users},).")

        valid_mask = (self.last_recs != -1)
        users_for_update = np.arange(self.n_users)[valid_mask]
        arms_for_update = self.last_recs[valid_mask]
        feedback_for_update = feedback[valid_mask]

        success_mask = (feedback_for_update == 1)
        failure_mask = (feedback_for_update == 0)

        self.successes[users_for_update[success_mask], arms_for_update[success_mask]] += 1.0
        self.failures[users_for_update[failure_mask], arms_for_update[failure_mask]] += 1.0
    
    def _get_current_scores(self) -> np.ndarray:
        if self.current_p_hat is not None:
            return self.current_p_hat
        else:
            return (self.successes + self.alpha_beta) / (self.successes + self.failures + 2 * self.alpha_beta + 1e-8)