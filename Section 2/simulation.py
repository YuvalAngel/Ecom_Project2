import numpy as np
import time

# Define TOTAL_TIME_LIMIT as it was in our previous discussion, or import it if defined elsewhere
TOTAL_TIME_LIMIT = 120 # seconds, or set to your desired value

class Simulation:
    def __init__(self, P: np.array, prices: np.array, budget: float, n_weeks: int, agent_class, agent_params: dict = None):
        self.P = P.copy()
        self.item_prices = prices.copy()
        self.budget = float(budget)
        self.n_weeks = n_weeks
        self.n_users, self.n_arms = P.shape

        self.agent_class = agent_class
        self.agent_params = agent_params if agent_params is not None else {}

    def _validate_recommendation(self, recommendation: np.ndarray) -> bool:
        """
        Validates the recommendation array. Returns True if valid, False otherwise.
        Prints critical errors, but less severe issues just cause a False return.
        """
        # 1. Check if it's a numpy array
        if not isinstance(recommendation, np.ndarray):
            print(f'CRITICAL ERROR: Recommendation is not an np.array (type: {type(recommendation)})')
            return False
        
        # 2. Check data type (should be integer-like)
        if not np.issubdtype(recommendation.dtype, np.integer):
            print(f'CRITICAL ERROR: Recommendation array elements are not integers (dtype: {recommendation.dtype})')
            return False
        
        # 3. Check shape
        if recommendation.shape != (self.n_users,):
            print(f'CRITICAL ERROR: Recommendation array has wrong shape ({recommendation.shape} vs expected ({self.n_users},))')
            return False
        
        # 4. Check if recommended item indices are within bounds or are the sentinel -1
        # An item index must be >= 0 and < n_arms, OR it must be -1 (for no recommendation)
        if not np.all(((recommendation >= 0) & (recommendation < self.n_arms)) | (recommendation == -1)):
            print(f'CRITICAL ERROR: Recommendation contains invalid item indices (not in [0, K-1] or -1): {recommendation}')
            return False

        # 5. Check budget constraint based on unique recommended items
        # Only process valid recommendations for budget check
        valid_recs_for_budget = recommendation[recommendation != -1]
        if valid_recs_for_budget.size > 0:
            # Use unique items to calculate total cost if multiple users get the same item
            unique_recommended_items = np.unique(valid_recs_for_budget)
            total_price = np.sum(self.item_prices[unique_recommended_items])
            if total_price > self.budget:
                # This is a validation failure, but not a "critical" Python error that crashes the interpreter
                # Just return False, the calling function will handle assigning 0 reward for this round.
                return False
            
        return True
    
    def run(self) -> int:
        """
        Runs a single simulation of the recommendation process for the configured agent.
        """
        total_time_taken = 0
        
        init_start = time.perf_counter()
        
        try:
            # Common parameters for all agents
            common_params = {
                "n_weeks": self.n_weeks,
                "n_users": self.n_users,
                "n_arms": self.n_arms,
                "prices": self.item_prices,
                "budget": self.budget,
            }
            agent = self.agent_class(
                **common_params,
                **self.agent_params
            )
        except Exception as e:
            print(f'Agent {self.agent_class.__name__} __init__ caused error: {e}. Returning 0 reward.')
            return 0 # Hard exit on initialization error
        
        init_end = time.perf_counter()
        total_time_taken += init_end - init_start
        
        cumulative_reward = 0
        
        for round_idx in range(self.n_weeks): # This loop *must* complete n_weeks times unless time limit hit
            round_start_time = time.perf_counter()
            
            # Default recommendations to -1 for all users in case of agent error
            recs = np.full(self.n_users, -1, dtype=int) 
            
            try:
                # Attempt to get recommendations from the agent
                agent_recs = agent.recommend() 
                if agent_recs is None:
                    print(f"WARNING: Agent {self.agent_class.__name__} returned None recommendation at round {round_idx}. Treating as no recommendations for this round.")
                    # recs remains np.full(self.n_users, -1) as default
                else:
                    recs = agent_recs # Use the agent's actual recommendation
            except Exception as e:
                print(f'ERROR: Agent {self.agent_class.__name__}.recommend() raised error at round {round_idx}: {e}. Treating as no recommendations for this round.')
                # recs remains np.full(self.n_users, -1) as default

            # --- Calculate Rewards and Prepare Feedback for Agent Update ---
            # Assume no valid recommendations and no rewards for this round initially
            feedback = np.zeros(self.n_users, dtype=int)

            if not self._validate_recommendation(recs):
                # If validation fails, it means the recommendations for this round are bad.
                # We penalize by assigning 0 reward for this round and skip update (feedback is already zeros).
                print(f'WARNING: Invalid recommendation from agent {self.agent_class.__name__} at round {round_idx}. Assigning 0 reward for this round.')
            else:
                # If recommendations are valid, calculate rewards normally for valid recommendations
                valid_mask = (recs != -1)
                
                if np.any(valid_mask): # Only process if there are valid recommendations
                    users_who_received_rec = np.arange(self.n_users)[valid_mask]
                    arms_recommended = recs[valid_mask]
                    
                    # Simulate probabilistic rewards based on P matrix
                    probs_for_recommended_arms = self.P[users_who_received_rec, arms_recommended]
                    rewards_for_recommended_arms = (np.random.rand(users_who_received_rec.size) < probs_for_recommended_arms).astype(int)
                    
                    # Populate the feedback array only for users who received a valid recommendation
                    feedback[users_who_received_rec] = rewards_for_recommended_arms
                # else: feedback remains all zeros if no valid recommendations were made (e.g., all -1)
            
            cumulative_reward += feedback.sum() # Sum rewards for this round
            
            # --- Handle Agent Update ---
            # Agent's update method now only needs the feedback array
            try:
                agent.update(feedback) 
            except Exception as e:
                print(f'ERROR: Agent {self.agent_class.__name__}.update() raised error at round {round_idx}: {e}. Continuing simulation.')
                # Crucially, we do NOT return here. Let the simulation continue.
            
            round_end_time = time.perf_counter()
            total_time_taken += (round_end_time - round_start_time)

            if total_time_taken > TOTAL_TIME_LIMIT:
                print(f'TOTAL TIME LIMIT EXCEEDED for {self.agent_class.__name__}. Terminating at round {round_idx+1}/{self.n_weeks}.')
                return cumulative_reward # Return whatever reward has been accumulated so far
        
        # If the loop completes all n_weeks without hitting time limit or critical error
        return cumulative_reward