# simulation.py

import time
import numpy as np
from tqdm import tqdm
from Recommender import Recommender
from test import test_1, test_2, test_3

TOTAL_TIME_LIMIT = 120  # seconds

class Simulation:
    def __init__(self, P: np.ndarray, prices: np.ndarray, budget: float, n_weeks: int):
        self.P = P.copy()
        self.item_prices = prices
        self.budget = budget
        self.n_weeks = n_weeks

    def _validate_recommendation(self, rec: np.ndarray) -> bool:
        if not isinstance(rec, np.ndarray) or not np.issubdtype(rec.dtype, np.integer):
            return False
        if rec.shape != (self.P.shape[0],) or ((rec < 0)|(rec >= self.P.shape[1])).any():
            return False
        cost = np.sum(self.item_prices[np.unique(rec)])
        return cost <= self.budget

    def simulate(self, smoothing: float, explore_rounds: int, max_iters: int) -> (float, float):
        total_time = 0.0
        rec = Recommender(
            n_weeks=self.n_weeks,
            n_users=self.P.shape[0],
            prices=self.item_prices,
            budget=self.budget,
            smoothing=smoothing,
            explore_rounds=explore_rounds,
            max_iters=max_iters
        )
        reward = 0.0

        for _ in range(self.n_weeks):
            start = time.perf_counter()
            picks = rec.recommend()
            mid = time.perf_counter()
            if not self._validate_recommendation(picks):
                return 0.0, total_time
            feedback = np.random.binomial(1, self.P[np.arange(self.P.shape[0]), picks])
            rec.update(feedback)
            end = time.perf_counter()

            total_time += (mid - start) + (end - mid)
            if total_time > TOTAL_TIME_LIMIT:
                break
            reward += feedback.sum()

        return reward, total_time

if __name__ == '__main__':
    # configs = [
    #     (0.5, 5, 100),
    #     (1.0, 5, 10),
    #     (1.0, 5, 50),
    #     (1.0, 5, 100),
    #     (1.0, 10, 50),
    #     (1.0, 20, 10),
    #     (1.0, 20, 20),
    #     (1.0, 20, 100)
    # ]
    configs = [
        (1.0, 20, 10),
        (1.0, 5, 100),
        (1.0, 5, 50),
        (1.0, 20, 20)
    ]

    trials = 50
    tests = [test_1, test_2, test_3]

    for i, test in enumerate(tests, start=1):
        print(f"\n=== Test {i} ===")
        for a, er, mi in configs:
            total_reward = 0.0
            total_time = 0.0
            print(f"Running a={a}, er={er}, mi={mi}...")
            for _ in tqdm(range(trials)):  # <-- Progress bar here
                sim = Simulation(test['P'], test['item_prices'], test['budget'], test['n_weeks'])
                r, t = sim.simulate(smoothing=a, explore_rounds=er, max_iters=mi)
                total_reward += r
                total_time += t
            avg_reward = total_reward / trials
            avg_time = total_time / trials
            print(f"a={a}, er={er}, mi={mi} -> Avg={avg_reward:.1f}, Time={avg_time:.2f}s")