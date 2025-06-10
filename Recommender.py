# recommender.py

import numpy as np


def build_feasible_set(prices: np.ndarray, budget: float, scores: np.ndarray) -> list:
    """
    Greedy feasible set builder selecting items with highest scores
    under budget constraints.

    Args:
        prices (np.ndarray): Cost per item.
        budget (float): Total budget.
        scores (np.ndarray): Scores per item to rank selection.

    Returns:
        list: Selected item indices within budget.
    """
    idx = np.argsort(-scores)
    S = []
    total_cost = 0.0
    for j in idx:
        if total_cost + prices[j] <= budget:
            S.append(j)
            total_cost += prices[j]
    return S



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

    def __hill__climb(self, S, p_hat):
        """Local search to improve set S under budget using add/remove/swap moves."""
        best_S = list(S)
        best_S_set = set(best_S)
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

            # Try removals (iterate over a copy to safely remove items)
            for k in best_S[:]:
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

            for k in best_S[:]:
                cand_S = [j for j in best_S if j != k]
                if cand_S:
                    candP = np.max(p_hat[:, cand_S], axis=1)
                    cand_cost = cost_S - self.costs[k]
                else:
                    candP = np.zeros(self.N)
                    cand_cost = 0.0
                gain = candP.sum()
                if gain > best_gain:
                    best_S.remove(k)
                    best_S_set.remove(k)
                    bestP = candP
                    best_gain = gain
                    cost_S = cand_cost
                    improved = True
                    break

            if not improved:
                break
            iters += 1

        return best_S


    def recommend(self) -> np.ndarray:
        self.round += 1
        
        if self.round <= self.explore_rounds:
            # create random scores for each user-item
            random_scores = np.random.rand(self.N, self.K)
            avg_scores = random_scores.mean(axis=0)
            S = build_feasible_set(self.costs, self.B, avg_scores)

            recs = np.random.choice(S, size=self.N)
            self.last_recs = recs
            return recs

        # Thompson Sampling: sample p_hat ~ Beta(successes + alpha, failures + alpha)
        p_hat = np.random.beta(self.successes + self.alpha, self.failures + self.alpha)

        # Or to use UCB instead, comment out above and uncomment below:
        # total = self.successes + self.failures + 2*self.alpha
        # mean = (self.successes + self.alpha) / (total + 1e-8)
        # confidence = np.sqrt(2 * np.log(self.round + 1) / (total + 1e-8))
        # p_hat = mean + confidence
        # p_hat = np.clip(p_hat, 0, 1)

        avg_scores = p_hat.mean(axis=0)
        best_S = build_feasible_set(self.costs, self.B, avg_scores)

        # hill-climb refine
        best_S = self.__hill__climb(best_S, p_hat)

        # assign to users
        p_hat_S = p_hat[:, best_S]               # shape (N, |best_S|)
        best_indices = np.argmax(p_hat_S, axis=1)  # index of best arm per user in best_S
        recs = np.array([best_S[i] for i in best_indices])
        self.last_recs = recs
        return recs

    def update(self, feedback: np.ndarray):
        success_mask = (feedback == 1)
        failure_mask = (feedback == 0)
        users = np.arange(self.N)

        self.successes[users[success_mask], self.last_recs[success_mask]] += 1.0
        self.failures[users[failure_mask], self.last_recs[failure_mask]] += 1.0


import numpy as np

class EpsilonGreedy:
    def __init__(self, n_users, n_arms, epsilon=0.1, prices=None, budget=None):
        self.n_users = n_users
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.prices = prices
        self.budget = budget
        self.counts = np.zeros((n_users, n_arms))
        self.values = np.zeros((n_users, n_arms))

    def recommend(self):
        if np.random.rand() < self.epsilon:
            scores = np.random.rand(self.n_arms)
        else:
            scores = self.values.mean(axis=0)

        S = build_feasible_set(self.prices, self.budget, scores)

        return np.array([max(S, key=lambda a: self.values[u, a]) for u in range(self.n_users)])


    def update(self, users, arms, rewards):
        for u, a, r in zip(users, arms, rewards):
            self.counts[u, a] += 1
            n = self.counts[u, a]
            v = self.values[u, a]
            self.values[u, a] = ((n - 1) / n) * v + (1 / n) * r




class UCB:
    def __init__(self, n_users, n_arms, prices, budget, c=1.0):
        self.n_users = n_users
        self.n_arms = n_arms
        self.prices = prices
        self.budget = budget
        self.c = c
        self.counts = np.zeros((n_users, n_arms))
        self.values = np.zeros((n_users, n_arms))
        self.total_counts = 1

    def recommend(self):
        self.total_counts += 1
        total = self.counts + 1e-8
        ucb = self.values + self.c * np.sqrt(np.log(self.total_counts) / total)

        scores = ucb.mean(axis=0)

        S = build_feasible_set(self.prices, self.budget, scores)

        return np.array([max(S, key=lambda a: ucb[u, a]) for u in range(self.n_users)])

    def update(self, users, arms, rewards):
        for u, a, r in zip(users, arms, rewards):
            self.counts[u, a] += 1
            n = self.counts[u, a]
            v = self.values[u, a]
            self.values[u, a] = ((n - 1) / n) * v + (1 / n) * r


class ThompsonSampling:
    def __init__(self, n_users, n_arms, prices, budget, alpha=0.1):
        self.n_users = n_users
        self.n_arms = n_arms
        self.prices = prices
        self.budget = budget
        self.successes = np.zeros((n_users, n_arms))
        self.failures = np.zeros((n_users, n_arms))
        self.alpha = alpha

    def recommend(self):
        samples = np.random.beta(self.successes + self.alpha,
                                 self.failures + self.alpha)

        scores = samples.mean(axis=0)

        S = build_feasible_set(self.prices, self.budget, scores)

        return np.array([max(S, key=lambda a: samples[u, a]) for u in range(self.n_users)])

    def update(self, users, arms, rewards):
        for u, a, r in zip(users, arms, rewards):
            if r > 0:
                self.successes[u, a] += 1
            else:
                self.failures[u, a] += 1


