from Models.BaseModel import *
import numpy as np

class EnsembleWeightedBandit(BaseBudgetedBandit):
    """
    An ensemble model for Budgeted Multi-Armed Bandits that combines the recommendations
    of multiple base bandit models using an Exponentially Weighted Average (Hedge-like) strategy.
    Weights of base models are updated based on their 'performance' relative to observed rewards.
    """
    def __init__(self, n_weeks: int, n_users: int, n_arms: int, prices: np.ndarray, budget: float,
                 base_models_and_configs: tuple, 
                 learning_rate: float = 0.1, 
                 initial_weight_value: float = 1.0):
        
        super().__init__(n_weeks, n_users, n_arms, prices, budget)

        self.base_models = []
        for ModelClass, config_frozenset in base_models_and_configs:
            config_dict = dict(config_frozenset) 
            model_instance = ModelClass(
                n_weeks=n_weeks, 
                n_users=n_users,
                n_arms=n_arms,
                prices=prices,
                budget=budget,
                **config_dict 
            )
            self.base_models.append(model_instance)
        
        self.num_base_models = len(self.base_models)
        self.learning_rate = learning_rate
        
        self.log_weights = np.full(self.num_base_models, np.log(initial_weight_value))
        
        self.last_round_model_scores = None 
        self.last_round_ensemble_recommendations = None 

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalizes scores for a given model to be between 0 and 1.
        Crucial for combining scores from models with different scales.
        """
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score == min_score:
            return np.ones_like(scores) * 0.5 
        return (scores - min_score) / (max_score - min_score)

    def recommend(self) -> np.ndarray:
        """
        Generates recommendations by aggregating scores from base models
        based on current ensemble weights and applying the budget constraint.
        """
        exp_log_weights = np.exp(self.log_weights)
        current_normalized_weights = exp_log_weights / np.sum(exp_log_weights)

        aggregated_scores = np.zeros((self.n_users, self.n_arms))
        all_model_scores_for_round = [] 

        for i, model in enumerate(self.base_models):
            model_scores = model._get_current_scores() 
            
            if model_scores.ndim == 1:
                model_scores = np.tile(model_scores, (self.n_users, 1))

            normalized_model_scores = self._normalize_scores(model_scores)
            
            aggregated_scores += current_normalized_weights[i] * normalized_model_scores
            all_model_scores_for_round.append(normalized_model_scores)

        self.last_round_model_scores = np.array(all_model_scores_for_round)

        feasible_set_S = build_feasible_set_generic(self.prices, self.budget, aggregated_scores, agg_fn=np.mean)

        recommendations = np.full(self.n_users, -1, dtype=int)
        if not feasible_set_S:
            self._store_recommendations(recommendations)
            return recommendations
            
        for u in range(self.n_users):
            chosen_arm = max(feasible_set_S, key=lambda a: aggregated_scores[u, a])
            recommendations[u] = chosen_arm
        
        self._store_recommendations(recommendations)
        self.last_round_ensemble_recommendations = recommendations
        return recommendations

    def update(self, feedback: np.ndarray):
        """
        Updates each base model with the observed feedback and then
        updates the ensemble weights based on base model performance relative to rewards.
        """
        for model in self.base_models:
            model.update(feedback) 
        
        if self.last_round_model_scores is None or self.last_round_ensemble_recommendations is None:
            return

        valid_indices = self.last_round_ensemble_recommendations != -1
        
        if np.any(valid_indices):
            for i in range(self.num_base_models):
                model_scores_for_round = self.last_round_model_scores[i]
                
                scores_of_chosen_arms = model_scores_for_round[np.arange(self.n_users), self.last_round_ensemble_recommendations]
                
                valid_scores = scores_of_chosen_arms[valid_indices]
                actual_feedback = feedback[valid_indices]

                if len(valid_scores) > 0:
                    model_reward_signal = np.sum(valid_scores * actual_feedback) 
                    self.log_weights[i] += self.learning_rate * model_reward_signal
                        
            exp_log_weights = np.exp(self.log_weights)
            self.weights = exp_log_weights / np.sum(exp_log_weights)
            
            self.weights = np.maximum(self.weights, 1e-6) # Add a small floor to weights
            self.weights = self.weights / np.sum(self.weights)

    def _get_current_scores(self) -> np.ndarray:
        """
        Returns the current aggregated and normalized scores for all user-arm pairs.
        """
        exp_log_weights = np.exp(self.log_weights)
        current_normalized_weights = exp_log_weights / np.sum(exp_log_weights)

        aggregated_scores = np.zeros((self.n_users, self.n_arms))
        for i, model in enumerate(self.base_models):
            model_scores = model._get_current_scores()
            if model_scores.ndim == 1:
                model_scores = np.tile(model_scores, (self.n_users, 1))
            aggregated_scores += current_normalized_weights[i] * self._normalize_scores(model_scores)
        return aggregated_scores