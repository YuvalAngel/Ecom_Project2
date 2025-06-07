# recommender.py

import numpy as np

class Recommender:
    def __init__(self,
                 n_weeks: int,
                 n_users: int,
                 prices: np.ndarray,
                 budget: float,
                 smoothing: float = 0.1,
                 explore_rounds: int = 10,
                 max_iters: int = 50):
        """
        Explore-Then-Exploit recommender under budget constraint with
        local hill-climbing and time-aware max_iters.

        Args:
            n_weeks       (int): Number of rounds (interface compatibility).
            n_users       (int): Number of users (N).
            prices        (np.ndarray[K,]): Cost per podcast.
            budget        (float): Budget per round (B).
            smoothing     (float): Prior pseudocount (α ≥ 0).
            explore_rounds(int): Number of initial pure exploration rounds.
            max_iters     (int): Max iterations for hill-climbing per recommend().
        """
        self.T = n_weeks
        self.N = n_users
        self.costs = np.array(prices, dtype=float)
        self.K = self.costs.size
        self.B = float(budget)
        self.alpha = float(smoothing)
        self.explore_rounds = int(explore_rounds)
        self.max_iters = int(max_iters)

        self.successes = np.zeros((self.N, self.K), dtype=float)
        self.failures  = np.zeros((self.N, self.K), dtype=float)
        self.last_recs = np.zeros(self.N, dtype=int)
        self.round = 0

    def _hill_climb(self, S, p_hat):
        """Local search to improve set S under budget using add/remove/swap moves."""
        best_S = list(S)
        bestP = np.max(p_hat[:, best_S], axis=1)
        best_gain = bestP.sum()
        iters = 0

        while iters < self.max_iters:
            improved = False
            cost_S = self.costs[best_S].sum()

            # try additions
            for k in range(self.K):
                if k in best_S or cost_S + self.costs[k] > self.B:
                    continue
                candP = np.maximum(bestP, p_hat[:, k])
                gain = candP.sum()
                if gain > best_gain:
                    best_S, bestP, best_gain = best_S + [k], candP, gain
                    improved = True
                    break
            if improved:
                iters += 1
                continue

            # try removals
            for k in best_S:
                cand_S = [j for j in best_S if j != k]
                candP = np.max(p_hat[:, cand_S], axis=1) if cand_S else np.zeros(self.N)
                gain = candP.sum()
                if gain > best_gain:
                    best_S, bestP, best_gain = cand_S, candP, gain
                    improved = True
                    break
            if improved:
                iters += 1
                continue

            # try swaps
            for i in best_S:
                for k in range(self.K):
                    if k in best_S:
                        continue
                    new_cost = cost_S - self.costs[i] + self.costs[k]
                    if new_cost > self.B:
                        continue
                    cand_S = [j for j in best_S if j != i] + [k]
                    candP = np.max(p_hat[:, cand_S], axis=1)
                    gain = candP.sum()
                    if gain > best_gain:
                        best_S, bestP, best_gain = cand_S, candP, gain
                        improved = True
                        break
                if improved:
                    break

            if not improved:
                break
            iters += 1

        return best_S

    def recommend(self) -> np.ndarray:
        self.round += 1
        p_hat = (self.successes + self.alpha) / (self.successes + self.failures + 2*self.alpha + 1e-8)

        # exploration
        if self.round <= self.explore_rounds:
            feasible = []
            for mask in range(1, 1 << self.K):
                cost = sum(self.costs[j] for j in range(self.K) if mask & (1 << j))
                if cost <= self.B:
                    feasible.append([j for j in range(self.K) if mask & (1 << j)])
            S = feasible[np.random.randint(len(feasible))]
            recs = np.random.choice(S, size=self.N)
            self.last_recs = recs
            return recs

        # exploitation: exhaustive search
        best_S, best_val = [], -np.inf
        for mask in range(1, 1 << self.K):
            total_cost = 0.0
            for j in range(self.K):
                if mask & (1 << j):
                    total_cost += self.costs[j]
                    if total_cost > self.B:
                        break
            else:
                val = sum(max(p_hat[n, j] for j in range(self.K) if mask & (1 << j))
                          for n in range(self.N))
                if val > best_val:
                    best_S, best_val = [j for j in range(self.K) if mask & (1 << j)], val

        # hill-climb refine
        best_S = self._hill_climb(best_S, p_hat)

        # assign to users
        recs = np.array([max(best_S, key=lambda k: p_hat[n, k]) for n in range(self.N)])
        self.last_recs = recs
        return recs

    def update(self, feedback: np.ndarray):
        for n, liked in enumerate(feedback):
            k = self.last_recs[n]
            if 0 <= k < self.K:
                if liked:
                    self.successes[n, k] += 1.0
                else:
                    self.failures[n, k] += 1.0
