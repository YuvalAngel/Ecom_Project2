import numpy as np

class Recommender:
    def __init__(self,
                 n_weeks: int,
                 n_users: int,
                 prices: np.ndarray,
                 budget: float,
                 smoothing: float = 1.0):
        """
        Online learning recommender under budget using a greedy knapsack approximation
        with Laplace-smoothed click-rate estimates per user-item.

        Args:
            n_weeks (int): Number of rounds (unused internally, but kept for interface compatibility).
            n_users (int): Number of users (N).
            prices (np.ndarray[K,]): Cost of each podcast.
            budget (float): Total budget per round (B).
            smoothing (float): Laplace smoothing parameter (default=1.0).
        """
        self.T = n_weeks
        self.N = n_users
        self.costs = np.array(prices, dtype=float)
        self.K = self.costs.size
        self.B = float(budget)
        self.smoothing = smoothing

        # Track successes and trials for each user-item pair
        self.successes = np.zeros((self.N, self.K), dtype=float)
        self.trials = np.zeros((self.N, self.K), dtype=float)

        # Last recommendations (for mapping feedback)
        self.last_recs = np.zeros(self.N, dtype=int)

    def recommend(self) -> np.ndarray:
        """
        Estimate click probabilities, solve a greedy knapsack for best subset S,
        then recommend to each user the item in S with highest estimated rate.

        Returns:
            np.ndarray of shape (N,) with item indices (or -1 if no feasible selection).
        """
        # Compute Laplace-smoothed estimates
        hatP = (self.successes + self.smoothing) / (self.trials + 2 * self.smoothing)

        # Greedy knapsack approximation: maximize sum_n max(hatP[n,k], bestP[n]) under cost
        bestP = np.zeros(self.N, dtype=float)
        chosen = []
        remaining = self.B
        candidates = set(range(self.K))
        while True:
            best_gain = 0.0
            best_k = None
            # Evaluate marginal gain per cost
            for k in candidates:
                cost = self.costs[k]
                if cost > remaining:
                    continue
                # Gain = sum over users of additional probability
                gain = np.sum(np.maximum(hatP[:, k] - bestP, 0.0))
                if gain <= 0:
                    continue
                ratio = gain / cost
                if ratio > best_gain:
                    best_gain = ratio
                    best_k = k
            if best_k is None:
                break
            # Accept item best_k
            chosen.append(best_k)
            remaining -= self.costs[best_k]
            # Update covered probabilities
            bestP = np.maximum(bestP, hatP[:, best_k])
            candidates.remove(best_k)

        # Form recommendations
        if not chosen:
            recs = -1 * np.ones(self.N, dtype=int)
        else:
            recs = np.zeros(self.N, dtype=int)
            for n in range(self.N):
                # pick the item with max hatP for user n among chosen
                recs[n] = int(max(chosen, key=lambda k: hatP[n, k]))

        self.last_recs = recs
        return recs

    def update(self, feedback: np.ndarray):
        """
        Update success/trial counts from binary feedback of last recommendations.

        Args:
            feedback (np.ndarray[N,]): 1 indicates success, 0 indicates failure.
        """
        for n, liked in enumerate(feedback):
            k = self.last_recs[n]
            if 0 <= k < self.K:
                self.trials[n, k] += 1.0
                if liked:
                    self.successes[n, k] += 1.0
