import numpy as np

class Recommender:
    def __init__(self,
                 n_weeks: int,
                 n_users: int,
                 prices: np.ndarray,
                 budget: float,
                 smoothing: float = 1.0):
        """
        Online learning recommender with exhaustive subset search under budget
        using Laplace-smoothed click-rate estimates.

        Args:
            n_weeks   (int): Number of rounds (for interface compatibility).
            n_users   (int): Number of users (N).
            prices    (np.ndarray[K,]): Cost of each podcast.
            budget    (float): Total budget per round (B).
            smoothing (float): Laplace smoothing parameter α>0.
        """
        self.T = n_weeks
        self.N = n_users
        self.costs = np.array(prices, dtype=float)
        self.K = self.costs.size
        self.B = float(budget)
        self.smoothing = float(smoothing)

        # Track successes/trials per (user, podcast)
        self.successes = np.zeros((self.N, self.K), dtype=float)
        self.trials    = np.zeros((self.N, self.K), dtype=float)

        # Remember last recommendations
        self.last_recs = np.zeros(self.N, dtype=int)

    def recommend(self) -> np.ndarray:
        """
        Estimate click probabilities and pick the subset S⊆{0..K-1} with cost≤B
        that maximizes total expected clicks via exhaustive search. Then recommend
        each user the item in S with highest estimated click-rate.

        Returns:
            recs (np.ndarray[N,]): podcast indices, or -1 if none fits.
        """
        # 1) Laplace-smoothed estimates: (successes+α)/(trials+2α)
        hatP = (self.successes + self.smoothing) / (self.trials + 2*self.smoothing)

        # 2) Exhaustive search over all subsets of [0..K): 2^K possibilities
        best_S = []
        best_val = -1.0
        K = self.K
        costs = self.costs
        B = self.B
        N = self.N

        # iterate masks from 1..2^K-1
        for mask in range(1, 1 << K):
            # compute total cost
            total_cost = 0.0
            # early prune: accumulate cost
            for j in range(K):
                if mask & (1 << j):
                    total_cost += costs[j]
                    if total_cost > B:
                        break
            else:
                # compute expected clicks: sum over users of max hatP[n,k] for k in subset
                val = 0.0
                for n in range(N):
                    max_p = 0.0
                    for j in range(K):
                        if mask & (1 << j):
                            p = hatP[n, j]
                            if p > max_p:
                                max_p = p
                    val += max_p
                # update best
                if val > best_val:
                    best_val = val
                    # store indices in S
                    best_S = [j for j in range(K) if (mask & (1 << j))]

        # 3) Form recommendations
        if not best_S:
            recs = -1 * np.ones(N, dtype=int)
        else:
            recs = np.zeros(N, dtype=int)
            for n in range(N):
                # choose k in best_S with max hatP[n,k]
                best_k = best_S[0]
                best_p = hatP[n, best_k]
                for k in best_S[1:]:
                    p = hatP[n, k]
                    if p > best_p:
                        best_k, best_p = k, p
                recs[n] = best_k

        self.last_recs = recs
        return recs

    def update(self, feedback: np.ndarray):
        """
        Update success/trial counts from binary feedback on last_recs.

        Args:
            feedback (np.ndarray[N,]): 1 if user clicked, else 0.
        """
        for n in range(self.N):
            k = self.last_recs[n]
            if 0 <= k < self.K:
                self.trials[n, k]   += 1.0
                if feedback[n]:
                    self.successes[n, k] += 1.0