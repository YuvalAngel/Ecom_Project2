
import numpy as np


def build_feasible_set_generic(prices: np.ndarray, budget: float, scores: np.ndarray, agg_fn=np.mean) -> list:
    """
    Generic greedy feasible set builder using customizable score aggregation
    and score-per-cost prioritization.

    Args:
        prices (np.ndarray): Cost per item (K,).
        budget (float): Total budget.
        scores (np.ndarray): Score matrix (N, K) or vector (K,).
        agg_fn (callable): Aggregation function over users (axis=0), e.g., np.mean, np.max.

    Returns:
        list: Selected item indices within budget.
    """
    if scores.ndim == 2:
        item_scores = agg_fn(scores, axis=0)
    else:
        item_scores = scores

    # Prioritize items based on score per unit of cost.
    # Add a small epsilon to prices to avoid division by zero.
    score_per_cost = item_scores / (prices + 1e-8)
    idx = np.argsort(-score_per_cost) # Sort in descending order of score-per-cost

    S, total_cost = [], 0.0
    for j in idx:
        if total_cost + prices[j] <= budget:
            S.append(j)
            total_cost += prices[j]
    return S


class THCR:
    """
    Recommender agent that combines Thompson Sampling with Hill Climbing optimization
    to select a feasible set of items maximizing expected user reward under a budget.

    Key Features:
    - Thompson Sampling estimates per-user click probabilities via Beta distributions.
    - Greedy feasible set construction followed by hill climbing for local improvement.
    - Supports multi-user selection under a shared cost constraint.

    Args:
        n_weeks (int): Total number of rounds (used for interface compatibility).
        n_users (int): Number of users in the environment.
        prices (np.ndarray): Array of item prices.
        budget (float): Budget constraint per round.
        smoothing (float): Pseudocount added to Beta priors (α).
        explore_rounds (int): Initial rounds of uniform exploration before learning.
        max_iters (int): Maximum number of hill climbing iterations per round.
    """
    def __init__(self,
                 n_weeks: int,
                 n_users: int,
                 prices: np.ndarray,
                 budget: float,
                 smoothing: float = 0.5,
                 explore_rounds: int = 50,
                 max_iters: int = 50):
        self.T = n_weeks
        self.N = n_users
        self.costs = np.array(prices, dtype=float)
        self.K = self.costs.size # Number of arms/items
        self.B = float(budget)
        self.alpha = float(smoothing)
        self.explore_rounds = int(explore_rounds)
        self.max_iters = int(max_iters)

        self.successes = np.zeros((self.N, self.K), dtype=float)
        self.failures = np.zeros((self.N, self.K), dtype=float)
        self.last_recs = np.full(self.N, -1, dtype=int)
        self.round = 0

    def __hill_climb(self, S: list, p_hat: np.ndarray) -> list:
        """Local search to improve set S under budget using add/remove/swap moves."""
        best_S = list(S)
        best_S_set = set(best_S)
        
        if not best_S:
            bestP = np.zeros(self.N)
            best_gain = 0.0
            cost_S = 0.0
        else:
            bestP = np.max(p_hat[:, best_S], axis=1)
            best_gain = bestP.sum()
            cost_S = self.costs[best_S].sum()

        iters = 0

        while iters < self.max_iters:
            improved = False

            # Try additions
            for k in range(self.K):
                if k in best_S_set or cost_S + self.costs[k] > self.B:
                    continue
                candP = np.maximum(bestP, p_hat[:, k])
                gain = candP.sum()
                if gain > best_gain:
                    best_S.append(k)
                    best_S_set.add(k)
                    bestP = candP
                    best_gain = gain
                    cost_S += self.costs[k]
                    improved = True
                    break
            if improved:
                iters += 1
                continue

            # Try removals
            for k in best_S[:]: # Iterate over a copy to safely remove items
                cand_S = [j for j in best_S if j != k]
                if cand_S:
                    candP = np.max(p_hat[:, cand_S], axis=1)
                else:
                    candP = np.zeros(self.N)
                gain = candP.sum()
                if gain > best_gain:
                    best_S.remove(k)
                    best_S_set.remove(k)
                    bestP = candP
                    best_gain = gain
                    cost_S -= self.costs[k]
                    improved = True
                    break
            if improved:
                iters += 1
                continue
            
            if not improved:
                break

        return best_S


    def recommend(self) -> np.ndarray:
        self.round += 1
        
        if self.round <= self.explore_rounds:
            random_scores = np.random.rand(self.N, self.K)
            S = build_feasible_set_generic(self.costs, self.B, random_scores, agg_fn=np.mean)

            if not S:
                recs = np.full(self.N, -1)
            else:
                recs = np.random.choice(S, size=self.N)
            
            self.last_recs = recs
            return recs

        # Thompson Sampling: sample p_hat ~ Beta(successes + alpha, failures + alpha)
        p_hat = np.random.beta(self.successes + self.alpha, self.failures + self.alpha)

        initial_S = build_feasible_set_generic(self.costs, self.B, p_hat, agg_fn=np.mean)

        best_S = self.__hill_climb(initial_S, p_hat)

        if not best_S:
            recs = np.full(self.N, -1)
        else:
            p_hat_S = p_hat[:, best_S]
            best_indices_in_S = np.argmax(p_hat_S, axis=1)
            recs = np.array([best_S[i] for i in best_indices_in_S])
        
        self.last_recs = recs
        return recs

    def update(self, feedback: np.ndarray):
        valid_mask = (self.last_recs != -1)
        users_for_update = np.arange(self.N)[valid_mask]
        arms_for_update = self.last_recs[valid_mask]
        feedback_for_update = feedback[valid_mask]

        success_mask = (feedback_for_update == 1)
        failure_mask = (feedback_for_update == 0)

        self.successes[users_for_update[success_mask], arms_for_update[success_mask]] += 1.0
        self.failures[users_for_update[failure_mask], arms_for_update[failure_mask]] += 1.0


class EpsilonGreedy:
    """
    Improved Epsilon-Greedy agent using UCB-based exploration and decaying ε schedule.

    During exploration, uses UCB-like bonus to guide exploration toward uncertain arms.
    Exploration probability decays over time to promote exploitation.

    Key Features:
    - Decaying ε with configurable minimum and decay rate.
    - UCB-style exploration scores during random exploration phase.
    - Supports multi-user selection with feasible set constraints.

    Args:
        n_users (int): Number of users.
        n_arms (int): Number of available arms/items.
        epsilon (float): Initial exploration probability.
        epsilon_min (float): Minimum allowed value of ε.
        decay (float): Multiplicative decay factor applied to ε after each update.
        prices (np.ndarray): Array of item prices.
        budget (float): Budget constraint per round.
    """
    def __init__(self, n_users: int, n_arms: int, prices: np.ndarray, budget: float,
                 epsilon: float = 0.3, epsilon_min: float = 0.01, decay: float = 0.99):
        self.n_users = n_users
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay = decay
        self.prices = np.array(prices, dtype=float)
        self.budget = float(budget)

        self.counts = np.full((self.n_users, self.n_arms), 1.0) 
        self.values = np.full((self.n_users, self.n_arms), 1.0)
        self.round_num = 0
        self.last_recs = np.full(self.n_users, -1, dtype=int)

    def recommend(self) -> np.ndarray:
        self.round_num += 1
        recommendations = []

        for u in range(self.n_users):
            chosen_arm = -1

            if np.random.rand() < self.epsilon:
                user_counts_safe = self.counts[u] + 1e-5 
                user_values = self.values[u]
                ucb_scores = user_values + np.sqrt(2 * np.log(self.round_num + 1) / user_counts_safe)

                S = build_feasible_set_generic(self.prices, self.budget, ucb_scores)
                
                if S:
                    chosen_arm = max(S, key=lambda a: ucb_scores[a])
            else:
                S = build_feasible_set_generic(self.prices, self.budget, self.values[u])

                if S:
                    chosen_arm = max(S, key=lambda a: self.values[u, a])
            
            recommendations.append(chosen_arm)
        
        self.last_recs = np.array(recommendations, dtype=int)
        return self.last_recs

    def update(self, users: np.ndarray, arms: np.ndarray, rewards: np.ndarray):
        valid_mask = (arms != -1)
        users_for_update = users[valid_mask]
        arms_for_update = arms[valid_mask]
        rewards_for_update = rewards[valid_mask]

        for u, a, r in zip(users_for_update, arms_for_update, rewards_for_update):
            self.counts[u, a] += 1
            n = self.counts[u, a]
            v = self.values[u, a]
            self.values[u, a] = ((n - 1) / n) * v + (1 / n) * r

        self.epsilon = max(self.epsilon * self.decay, self.epsilon_min)
        

class UCB:
    """
    Upper Confidence Bound (UCB) agent for budget-constrained multi-user recommendation.
    Utilizes Optimistic Initialization for enhanced exploration and cold-start handling.

    Selects arms using a UCB strategy that explicitly uses Bernoulli variance,
    balancing exploitation of known high-value arms with exploration of uncertain ones.

    Key Features:
    - Optimistic initialization of mean rewards and counts for implicit exploration.
    - UCB scores computed per user, incorporating empirical mean and Bernoulli variance.
    - Feasible set built using average UCB scores.
    - Deterministic arm selection based on max UCB value.

    Args:
        n_users (int): Number of users.
        n_arms (int): Number of arms/items to choose from.
        prices (np.ndarray): Array of item prices.
        budget (float): Budget constraint per round.
        c (float): Exploration coefficient for the UCB confidence bound.
                   Commonly 1.0 or sqrt(2).
    """
    def __init__(self, n_users: int, n_arms: int, prices: np.ndarray, budget: float, 
                 c: float = 1.0): # Renamed c_explore to c
        self.n_users = n_users
        self.n_arms = n_arms
        self.prices = np.array(prices, dtype=float)
        self.budget = float(budget)
        self.c = c # Renamed self.c_explore to self.c        

        # Optimistic Initialization:
        # Initialize values to 1.0 (max reward) and counts to 1.0 (one pseudo-play)
        # This makes unplayed arms appear very attractive initially, encouraging exploration.
        self.counts = np.full((self.n_users, self.n_arms), 1.0) 
        self.values = np.full((self.n_users, self.n_arms), 1.0) 
        
        self.round_num = 0 # Global round counter

        # The similarity matrix is a non-standard addition to UCB, but retained from your original code.
        self.similarity = np.eye(n_users) * 0.8 + 0.2 / n_users 
        
        self.last_recs = np.full(self.n_users, -1, dtype=int)


    def recommend(self) -> np.ndarray:
        self.round_num += 1 # Increment global round counter

        # UCB phase (runs from the first round due to optimistic initialization):
        
        # Add a small epsilon to counts in case of actual 0 counts (though optimistic init helps)
        counts_safe = self.counts + 1e-8 

        # Empirical variance for Bernoulli rewards: p * (1 - p)
        # Use np.clip to ensure values are within [0, 1] for variance calculation stability.
        clipped_values = np.clip(self.values, 0, 1)
        empirical_variance = clipped_values * (1 - clipped_values)
        
        # UCB formula: mean + c * sqrt(2 * variance * ln(T) / n_i)
        log_term = np.log(self.round_num + 1e-8) # Add epsilon to log argument for stability

        ucb_raw = (
            self.values 
            + self.c * np.sqrt(2 * empirical_variance * log_term / counts_safe) # Using self.c
        )
        
        # Apply similarity smoothing and normalization.
        ucb = (self.similarity @ ucb_raw) / self.similarity.sum(axis=1, keepdims=True) 

        # Build the feasible set using build_feasible_set_generic, averaging UCB scores.
        S = build_feasible_set_generic(self.prices, self.budget, ucb, agg_fn=np.mean)

        recommendations = []
        if not S:
            recommendations = np.full(self.n_users, -1)
        else:
            for u in range(self.n_users):
                chosen_arm = max(S, key=lambda a: ucb[u, a])
                recommendations.append(chosen_arm)
        
        self.last_recs = np.array(recommendations, dtype=int)
        return self.last_recs

    def update(self, users: np.ndarray, arms: np.ndarray, rewards: np.ndarray):
        # Update counts and values only for valid recommendations.
        valid_mask = (arms != -1)
        users_for_update = users[valid_mask]
        arms_for_update = arms[valid_mask]
        rewards_for_update = rewards[valid_mask]

        for u, a, r in zip(users_for_update, arms_for_update, rewards_for_update):
            # Update counts: Increment from the initial 1.0 pseudo-count
            self.counts[u, a] += 1
            n = self.counts[u, a]
            
            # Update mean reward for Bernoulli rewards
            v_old = self.values[u, a]
            self.values[u, a] = ((n - 1) / n) * v_old + (1 / n) * r


class ThompsonSampling:
    """
    Thompson Sampling agent for multi-user recommendation with budget constraints.

    Samples click-through probabilities from per-arm Beta distributions and uses these
    to construct a feasible set and assign arms to users with maximum sampled reward.

    Key Features:
    - Bayesian strategy using flexible Beta priors for click probability estimation.
    - Posterior updates based on binary feedback (success/failure).
    - Budget-aware feasible set construction per round.

    Args:
        n_users (int): Number of users.
        n_arms (int): Number of available arms/items.
        prices (np.ndarray): Array of item prices.
        budget (float): Budget constraint per round.
        alpha_prior (float): Prior 'successes' count for Beta distributions (alpha parameter).
                             Default 1.0 for uniform prior.
        beta_prior (float): Prior 'failures' count for Beta distributions (beta parameter).
                            Default 1.0 for uniform prior.
    """
    def __init__(self, n_users: int, n_arms: int, prices: np.ndarray, budget: float, 
                 alpha_prior: float = 1.0, beta_prior: float = 1.0): # Changed hyperparameters
        self.n_users = n_users
        self.n_arms = n_arms
        self.prices = np.array(prices, dtype=float)
        self.budget = float(budget)
        self.successes = np.zeros((self.n_users, self.n_arms), dtype=float)
        self.failures = np.zeros((self.n_users, self.n_arms), dtype=float)
        self.alpha_prior = alpha_prior # New hyperparameter
        self.beta_prior = beta_prior   # New hyperparameter
        self.last_recs = np.full(self.n_users, -1, dtype=int)

    def recommend(self) -> np.ndarray:
        # Sample click-through probabilities from the Beta posterior for each user-arm pair.
        # The Beta distribution parameters are (actual_successes + prior_alpha) and (actual_failures + prior_beta).
        samples = np.random.beta(self.successes + self.alpha_prior, self.failures + self.beta_prior)

        # Build the feasible set using the average of sampled probabilities across users.
        S = build_feasible_set_generic(self.prices, self.budget, samples, agg_fn=np.mean)

        recommendations = []
        if not S:
            recommendations = np.full(self.n_users, -1)
        else:
            for u in range(self.n_users):
                # For each user, choose the arm from the feasible set S that has the maximum sampled value for them.
                chosen_arm = max(S, key=lambda a: samples[u, a])
                recommendations.append(chosen_arm)
        
        self.last_recs = np.array(recommendations, dtype=int)
        return self.last_recs

    def update(self, users: np.ndarray, arms: np.ndarray, rewards: np.ndarray):
        # Filter out invalid recommendations (-1 arms)
        valid_mask = (arms != -1)
        users_for_update = users[valid_mask]
        arms_for_update = arms[valid_mask]
        rewards_for_update = rewards[valid_mask]

        # Update successes and failures based on binary rewards (0 or 1)
        for u, a, r in zip(users_for_update, arms_for_update, rewards_for_update):
            if r > 0: # Reward is 1 (success)
                self.successes[u, a] += 1
            else: # Reward is 0 (failure)
                self.failures[u, a] += 1