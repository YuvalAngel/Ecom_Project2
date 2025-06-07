import time
from test import test_1, test_2, test_3

from Recommender import Recommender
import numpy as np

TOTAL_TIME_LIMIT = 120  # seconds

class Simulation():
    def __init__(self, P: np.array, prices: np.array, budget, n_weeks: int):
        self.P = P.copy()
        self.item_prices = prices
        self.budget = budget
        self.n_weeks = n_weeks

    def _validate_recommendation(self, recommendation):
        if not isinstance(recommendation, np.ndarray):
            return False
        if not np.issubdtype(recommendation.dtype, np.integer):
            return False
        if recommendation.shape != (self.P.shape[0],):
            return False
        if ((recommendation < 0) | (recommendation >= self.P.shape[1])).any():
            return False
        podcasts = np.unique(recommendation)
        total_price = np.sum(self.item_prices[podcasts])
        if total_price > self.budget:
            return False
        return True

    def simulate(self, smoothing=0.1, explore_rounds=10):
        total_time_taken = 0

        # Use the wrapper here
        recommender = Recommender(
            n_weeks=self.n_weeks,
            n_users=self.P.shape[0],
            prices=self.item_prices.copy(),
            budget=self.budget,
            smoothing=smoothing,
            explore_rounds=explore_rounds
        )

        reward = 0

        for round_idx in range(self.n_weeks):
            start = time.perf_counter()
            recommendation = recommender.recommend()
            end = time.perf_counter()

            if recommendation is None or not self._validate_recommendation(recommendation):
                return 0, total_time_taken

            results = np.random.binomial(n=1, p=self.P[np.arange(self.P.shape[0]), recommendation])
            reward += np.sum(results)

            update_start = time.perf_counter()
            recommender.update(results)
            update_end = time.perf_counter()

            round_time = (end - start) + (update_end - update_start)
            if total_time_taken + round_time > TOTAL_TIME_LIMIT:
                return reward, total_time_taken
            total_time_taken += round_time

        return reward, total_time_taken



if __name__ == '__main__':
    for test_number, test in enumerate([test_1, test_2, test_3], start=1):
        print(f"\n=== Test {test_number} ===")

        trials_per_config = 5
        final_trials = 20

        best_config = None
        best_avg_reward = -1

        explore_rounds_options = [0, 5, 10, 20]
        smoothing = 1.0

        for explore_rounds in explore_rounds_options:
            total_reward = 0
            for _ in range(trials_per_config):
                sim = Simulation(test['P'], test['item_prices'], test['budget'], test['n_weeks'])
                reward, _ = sim.simulate(smoothing=smoothing, explore_rounds=explore_rounds)
                total_reward += reward
            avg_reward = total_reward / trials_per_config
            print(f"Config: smoothing={smoothing}, explore_rounds={explore_rounds}, Avg Reward={avg_reward:.2f}")

            # Track best config here
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_config = (smoothing, explore_rounds)

        print(f"\nBest Config for Test {test_number}: smoothing={best_config[0]}, explore_rounds={best_config[1]}")

        # Final run on best config
        final_total_reward = 0
        final_total_time = 0
        for _ in range(final_trials):
            sim = Simulation(test['P'], test['item_prices'], test['budget'], test['n_weeks'])
            reward, run_time = sim.simulate(smoothing=best_config[0], explore_rounds=best_config[1])
            final_total_reward += reward
            final_total_time += run_time

        final_avg_reward = final_total_reward / final_trials
        avg_time = final_total_time / final_trials

        print(f"Final Avg Reward: {final_avg_reward:.2f}")
        print(f"Avg Time Per Run: {avg_time:.2f} seconds")
