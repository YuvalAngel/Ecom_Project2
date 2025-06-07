import numpy as np

class Recommender:
    def __init__(self, n_weeks: int, n_users: int, prices: np.array, budget: int):
        """
        Initializes the recommendation system.

        Args:
            n_weeks (int): Number of weeks to run.
            n_users (int): Number of users.
            prices (np.array): Array of podcast production costs (length K).
            budget (int): Weekly budget constraint.
        """
        self.n_rounds = n_weeks
        self.n_users = n_users
        self.item_prices = prices
        self.budget = budget

        self.K = len(prices)
        self.feedback = np.zeros((n_users, self.K))  # Likes count per user and podcast
        self.counts = np.ones((n_users, self.K))     # To avoid division by zero
        self.week = 0

    def recommend(self) -> np.array:
        """
        Recommend a podcast to each user for the current week.

        Returns:
            np.array: An array of shape (n_users,) where each entry is a podcast index
                      that was recommended to the corresponding user.
        """
        self.week += 1

        avg_rewards = self.feedback / self.counts
        global_estimates = avg_rewards.mean(axis=0)

        # Budgeted greedy selection
        value_per_cost = global_estimates / self.item_prices
        sorted_indices = np.argsort(-value_per_cost)

        produced = np.zeros(self.K, dtype=bool)
        total_cost = 0
        for i in sorted_indices:
            if total_cost + self.item_prices[i] <= self.budget:
                produced[i] = True
                total_cost += self.item_prices[i]

        recommendations = np.zeros(self.n_users, dtype=int)
        for user in range(self.n_users):
            personal_score = self.feedback[user] / self.counts[user]
            masked_score = personal_score * produced
            if np.any(produced):
                if np.all(masked_score == 0):
                    recommendations[user] = np.random.choice(np.where(produced)[0])
                else:
                    recommendations[user] = np.argmax(masked_score)
            else:
                recommendations[user] = 0

        self.last_recommendations = recommendations
        return recommendations

    def update(self, results: np.array):
        """
        Update feedback based on user responses.

        Args:
            results (np.array): A binary array of shape (n_users,) indicating user feedback
                                (1 if user liked the podcast, 0 otherwise).
        """
        for user_id, liked in enumerate(results):
            podcast_id = self.last_recommendations[user_id]
            self.feedback[user_id, podcast_id] += liked
            self.counts[user_id, podcast_id] += 1
