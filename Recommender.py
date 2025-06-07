import numpy as np

class Recommender:
    def __init__(self,
                 n_weeks: int,
                 n_users: int,
                 prices: np.ndarray,
                 budget: float,
                 smoothing: float = 0.1,  # smaller smoothing or even zero
                 explore_rounds: int = 10):
        """
        Explore-Then-Explore recommender under budget constraint.

        Args:
            n_weeks       (int): Number of rounds (interface compatibility).
            n_users       (int): Number of users (N).
            prices        (np.ndarray[K,]): Cost per podcast.
            budget        (float): Budget per round (B).
            smoothing     (float): Prior pseudocount (α ≥ 0).
            explore_rounds(int): Number of initial pure exploration rounds (random).
        """
        self.T = n_weeks
        self.N = n_users
        self.costs = np.array(prices, dtype=float)
        self.K = self.costs.size
        self.B = float(budget)
        self.alpha = float(smoothing)
        self.explore_rounds = int(explore_rounds)

        self.successes = np.zeros((self.N, self.K), dtype=float)
        self.failures  = np.zeros((self.N, self.K), dtype=float)
        self.last_recs = np.zeros(self.N, dtype=int)
        self.round = 0

    def recommend(self) -> np.ndarray:
        self.round += 1

        # Calculate empirical means with smoothing
        p_hat = (self.successes + self.alpha) / (self.successes + self.failures + 2 * self.alpha)

        if self.round <= self.explore_rounds:
            # Random exploration: pick random feasible subset
            feasible_subsets = []
            attempts = 0
            max_attempts = 1000
            while attempts < max_attempts:
                mask = np.random.randint(1, 1 << self.K)
                total_cost = 0.0
                for j in range(self.K):
                    if mask & (1 << j):
                        total_cost += self.costs[j]
                        if total_cost > self.B:
                            break
                else:
                    feasible_subsets.append([j for j in range(self.K) if (mask & (1 << j))])
                if feasible_subsets:
                    break
                attempts += 1

            if not feasible_subsets:
                recs = -1 * np.ones(self.N, dtype=int)
            else:
                S = feasible_subsets[0]
                recs = np.zeros(self.N, dtype=int)
                for n in range(self.N):
                    recs[n] = np.random.choice(S)

            self.last_recs = recs
            return recs

        # Exploitation: maximize expected success sum over best feasible subset
        best_S = []
        best_val = -np.inf
        for mask in range(1, 1 << self.K):
            total_cost = 0.0
            for j in range(self.K):
                if mask & (1 << j):
                    total_cost += self.costs[j]
                    if total_cost > self.B:
                        break
            else:
                val = 0.0
                for n in range(self.N):
                    max_p = 0.0
                    for j in range(self.K):
                        if mask & (1 << j) and p_hat[n, j] > max_p:
                            max_p = p_hat[n, j]
                    val += max_p
                if val > best_val:
                    best_val = val
                    best_S = [j for j in range(self.K) if (mask & (1 << j))]

        if not best_S:
            recs = -1 * np.ones(self.N, dtype=int)
        else:
            recs = np.zeros(self.N, dtype=int)
            for n in range(self.N):
                best_k = best_S[0]
                best_p = p_hat[n, best_k]
                for k in best_S[1:]:
                    if p_hat[n, k] > best_p:
                        best_k, best_p = k, p_hat[n, k]
                recs[n] = best_k

        self.last_recs = recs
        return recs

    def update(self, feedback: np.ndarray):
        for n in range(self.N):
            k = self.last_recs[n]
            if 0 <= k < self.K:
                if feedback[n]:
                    self.successes[n, k] += 1.0
                else:
                    self.failures[n, k] += 1.0




