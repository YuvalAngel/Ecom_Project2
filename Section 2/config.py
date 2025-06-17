from Models import *

import numpy as np # Ensure numpy is imported if not already in your file


base_configs = {

    # THCR: [
    #     # Varying smoothing with current best explore_rounds = 10
    #     # {'smoothing': 0.05, 'explore_rounds': 5}, # Current best
    #     {'smoothing': 0.05, 'explore_rounds': 10},
    #     {'smoothing': 0.05, 'explore_rounds': 20},

    #     # {'smoothing': 0.10, 'explore_rounds': 10},
    # #     {'smoothing': 0.15, 'explore_rounds': 10},
    # ],


    # THCR: [
    #     # Current best 14500
    #     {'smoothing': 0.05, 'explore_rounds': 10, 'n_hill_climb_restarts': 5, 'random_walk_prob': 0.05, 'initial_set_noise_scale': 0.05},
        
    #     {'smoothing': 0.05, 'explore_rounds': 10, 'n_hill_climb_restarts': 10, 'random_walk_prob': 0.05, 'initial_set_noise_scale': 0.05},
    #     {'smoothing': 0.05, 'explore_rounds': 10, 'n_hill_climb_restarts': 10, 'random_walk_prob': 0.05, 'initial_set_noise_scale': 0.05},
    # ],

    UCB: [
        {'c': 0.01}, # Current best
        {'c': 0.005},
        {'c': 0.001},
        # {'c': 0.02},
        # {'c': 0.05},
        # {'c': 0.1},
        {'c': 0.5},
        {'c': 1.0},
    ],

    ThompsonSampling: [
        {'alpha_prior': 0.1, 'beta_prior': 2.0}, # Current best
        {'alpha_prior': 0.1, 'beta_prior': 1.0},
        {'alpha_prior': 0.05, 'beta_prior': 1.0},
    ],

    EpsilonGreedy: [
        {'epsilon': 0.10, 'decay': 0.98},

        {'epsilon': 0.20, 'decay': 0.97},

        {'epsilon': 0.25, 'decay': 0.95},
    ],

    GreedyBudget: [
        {'initial_q_value': 1.0, 'initial_counts': 1}, # Default optimistic initialization
        {'initial_q_value': 0.5, 'initial_counts': 1}, # Neutral initialization
    ],

    SoftmaxBudget: [
        {'temperature': 1.0, 'decay_rate': 0.99, 'min_temperature': 0.05, 'initial_q_value': 1.0, 'initial_counts': 1},
        {'temperature': 0.5, 'decay_rate': 0.995, 'min_temperature': 0.1, 'initial_q_value': 1.0, 'initial_counts': 1},
        {'temperature': 2.0, 'decay_rate': 0.98, 'min_temperature': 0.01, 'initial_q_value': 1.0, 'initial_counts': 1},
    ],

    ExploreThenCommitBudget: [
        {'explore_rounds': 50, 'initial_q_value': 1.0, 'initial_counts': 1},
        {'explore_rounds': 100, 'initial_q_value': 1.0, 'initial_counts': 1},
        {'explore_rounds': 200, 'initial_q_value': 1.0, 'initial_counts': 1},
    ],
    LinUCB: [
        {
            'alpha': 0.1,         # Lower exploration to test if features guide learning
            'lambda_reg': 0.1,    # Standard regularization strength
        },
        {
            'alpha': 0.5,         # Moderate exploration
            'lambda_reg': 0.1,
        },
        {
            'alpha': 1.0,         # Higher exploration
            'lambda_reg': 0.01,   # Less regularization (can lead to faster learning but more overfitting)
        },
    ],
    ThompsonSamplingSimilarity: [
        {
            # Using the default 'block' similarity generation
            'alpha_prior': 1.0, 
            'beta_prior': 1.0,    
            'default_sim_type': 'block', # Explicitly specify type
            'sim_group_size': 5,         # How big are the user groups
            'sim_self_weight': 0.7       # How much each user weighs themselves
        },
        {
            # Another configuration, trying a different default type
            'alpha_prior': 0.5, 
            'beta_prior': 0.5,    
            'default_sim_type': 'uniform' # All users equally similar
        },
        {
            # If you want to use an identity matrix (no smoothing) by default
            'alpha_prior': 2.0, 
            'beta_prior': 2.0,
            'default_sim_type': 'identity' 
        },
    ],
    MF_UCB: [
        {
            'latent_dim': 5,      # Smaller latent dimension
            'learning_rate': 0.05,
            'alpha': 0.5          # Moderate exploration
        },
        {
            'latent_dim': 10,     # Larger latent dimension (can capture more complexity)
            'learning_rate': 0.01, # Slower learning rate for stability
            'alpha': 0.8          # Higher exploration
        },
        {
            'latent_dim': 5,
            'learning_rate': 0.1, # Faster learning rate
            'alpha': 0.2          # Lower exploration (more exploitation)
        },
    ],

    UCB_MB: [
        {'c': 0.005, 'initial_q_value': 1.0, 'initial_counts': 1}, # New: Even lower c, closer to previous best
        {'c': 0.01, 'initial_q_value': 1.0, 'initial_counts': 1},
        {'c': 0.02, 'initial_q_value': 1.0, 'initial_counts': 1}, # New: Slightly higher c
        {'c': 0.05, 'initial_q_value': 1.0, 'initial_counts': 1},
        {'c': 0.1, 'initial_q_value': 1.0, 'initial_counts': 1},
        {'c': 0.01, 'initial_q_value': 0.5, 'initial_counts': 1}, # New: Previous best c with less optimistic start
        {'c': 0.05, 'initial_q_value': 0.5, 'initial_counts': 1},
    ],

    BudgetedThompsonSampling: [
        {'alpha_prior': 1.0, 'beta_prior': 1.0},
        {'alpha_prior': 0.5, 'beta_prior': 0.5},
        {'alpha_prior': 2.0, 'beta_prior': 2.0},
        {'alpha_prior': 0.1, 'beta_prior': 1.0},
        {'alpha_prior': 1.0, 'beta_prior': 0.1},
        {'alpha_prior': 0.05, 'beta_prior': 1.0}, # New: Even more aggressive exploration for less enjoyed arms
        {'alpha_prior': 0.1, 'beta_prior': 0.5}, # New: Slightly less pessimistic initial view but still low beta
    ],

    GreedyCostEfficiency: [
        {'explore_rounds': 0, 'initial_q_value': 1.0, 'initial_counts': 1, 'exploitation_noise_scale': 0.0}, # New: Pure greedy (no noise), optimistic
        {'explore_rounds': 0, 'initial_q_value': 0.5, 'initial_counts': 1, 'exploitation_noise_scale': 0.0}, # New: Pure greedy (no noise), balanced
        {'explore_rounds': 0, 'initial_q_value': 1.0, 'initial_counts': 1, 'exploitation_noise_scale': 0.0001}, # Original best with minimal noise
        {'explore_rounds': 0, 'initial_q_value': 1.0, 'initial_counts': 1, 'exploitation_noise_scale': 0.0005}, # New: Slightly more noise for optimistic pure greedy
        {'explore_rounds': 0, 'initial_q_value': 1.0, 'initial_counts': 1, 'exploitation_noise_scale': 0.001},
        {'explore_rounds': 10, 'initial_q_value': 0.5, 'initial_counts': 1, 'exploitation_noise_scale': 0.001},
        {'explore_rounds': 50, 'initial_q_value': 0.5, 'initial_counts': 1, 'exploitation_noise_scale': 0.001},
        {'explore_rounds': 50, 'initial_q_value': 0.5, 'initial_counts': 1, 'exploitation_noise_scale': 0.005},
        {'explore_rounds': 50, 'initial_q_value': 1.0, 'initial_counts': 1, 'exploitation_noise_scale': 0.001}, # New: Moderate exploration, optimistic, slight noise
    ],

    FractionalKnapsackDecreasingEpsilonGreedy: [
        {'initial_epsilon': 1.0, 'epsilon_decay': 0.99, 'min_epsilon': 0.05, 'initial_q_value': 0.5, 'initial_counts': 1},
        {'initial_epsilon': 0.8, 'epsilon_decay': 0.98, 'min_epsilon': 0.1, 'initial_q_value': 0.5, 'initial_counts': 1},
        {'initial_epsilon': 0.5, 'epsilon_decay': 0.995, 'min_epsilon': 0.01, 'initial_q_value': 1.0, 'initial_counts': 1},
        {'initial_epsilon': 0.7, 'epsilon_decay': 0.99, 'min_epsilon': 0.05, 'initial_q_value': 0.5, 'initial_counts': 1}, # New: Slightly adjusted decay
    ],

    AdaptiveBudgetCombinatorialBandit: [
        {'alpha': 0.1, 'initial_q_value': 1.0, 'initial_counts': 1},
        {'alpha': 0.5, 'initial_q_value': 1.0, 'initial_counts': 1},
        {'alpha': 1.0, 'initial_q_value': 1.0, 'initial_counts': 1},
        {'alpha': 0.2, 'initial_q_value': 0.5, 'initial_counts': 1},
        {'alpha': 0.15, 'initial_q_value': 0.5, 'initial_counts': 1}, # New: Fine-tune alpha around previous best
        {'alpha': 0.25, 'initial_q_value': 0.5, 'initial_counts': 1}, # New: Fine-tune alpha around previous best
    ],
    CBwKUCBV: [
        # Top performer from previous run (for direct comparison and confirmation)
        {'alpha': 0.050, 'epsilon': 1e-6, 'add_noise_to_knapsack': True, 'knapsack_noise_scale': 1e-4}, # Effectively no noise, based on previous result

        # Explore alpha around 0.05
        {'alpha': 0.040, 'epsilon': 1e-6, 'add_noise_to_knapsack': True, 'knapsack_noise_scale': 1e-4},
        {'alpha': 0.060, 'epsilon': 1e-6, 'add_noise_to_knapsack': True, 'knapsack_noise_scale': 1e-4},

        # Investigate knapsack noise with best alpha (0.05)
        {'alpha': 0.050, 'epsilon': 1e-6, 'add_noise_to_knapsack': True, 'knapsack_noise_scale': 1e-5}, # Very small noise
        {'alpha': 0.050, 'epsilon': 1e-6, 'add_noise_to_knapsack': True, 'knapsack_noise_scale': 1e-6}, # As previously
        {'alpha': 0.050, 'epsilon': 1e-6, 'add_noise_to_knapsack': True, 'knapsack_noise_scale': 1e-3}, # Higher noise (as suggested)
        {'alpha': 0.050, 'epsilon': 1e-6, 'add_noise_to_knapsack': True, 'knapsack_noise_scale': 1e-2}, # Even higher noise (as suggested)

        # Test alpha 0.1 with optimal knapsack noise (or lack thereof)
        {'alpha': 0.100, 'epsilon': 1e-6, 'add_noise_to_knapsack': False, 'knapsack_noise_scale': 0.0}, # Second best overall previously

        # Test alpha 0.1 with a low noise scale
        {'alpha': 0.100, 'epsilon': 1e-6, 'add_noise_to_knapsack': True, 'knapsack_noise_scale': 1e-4},

        # Re-test alpha 0.2 with no noise (just to confirm its poor performance, or see if knapsack noise was the issue)
        {'alpha': 0.200, 'epsilon': 1e-6, 'add_noise_to_knapsack': True, 'knapsack_noise_scale': 0.0},
    ],
    CBwKTunedUCB: [
        # Best performers from previous run
        {'epsilon': 1e-6, 'add_noise_to_knapsack': False, 'knapsack_noise_scale': 0.0}, # Previously best
        {'epsilon': 1e-6, 'add_noise_to_knapsack': True, 'knapsack_noise_scale': 0.0},  # Second best

        # Explore knapsack noise more broadly
        {'epsilon': 1e-6, 'add_noise_to_knapsack': True, 'knapsack_noise_scale': 1e-5},
        {'epsilon': 1e-6, 'add_noise_to_knapsack': True, 'knapsack_noise_scale': 1e-4}, # Existing tested value
        {'epsilon': 1e-6, 'add_noise_to_knapsack': True, 'knapsack_noise_scale': 5e-4}, # Existing tested value
        {'epsilon': 1e-6, 'add_noise_to_knapsack': True, 'knapsack_noise_scale': 1e-3}, # As suggested
        {'epsilon': 1e-6, 'add_noise_to_knapsack': True, 'knapsack_noise_scale': 1e-2}, # As suggested (very high)

        # Test with slightly larger epsilon (less common but worth one try)
        {'epsilon': 1e-3, 'add_noise_to_knapsack': True, 'knapsack_noise_scale': 1e-4}, # As suggested

        # Two more with high noise to see if it helps escape any local knapsack issues
        {'epsilon': 1e-6, 'add_noise_to_knapsack': True, 'knapsack_noise_scale': 2e-3},
        {'epsilon': 1e-6, 'add_noise_to_knapsack': True, 'knapsack_noise_scale': 5e-3},
    ],

    CBwKGreedyUCB: [
        # These are all good
        {'alpha': 0.18},
        {'alpha': 0.15},
        {'alpha': 0.20},
    ],
    
    THCR: [
        # 1. Overall Best (Reward: 14444.90)
        {'smoothing': 0.050, 'explore_rounds': 10, 'n_hill_climb_restarts': 10, 'random_walk_prob': 0.150, 'initial_set_noise_scale': 0.050},
        
        # 2. Second Best (Reward: 14439.40)
        {'smoothing': 0.050, 'explore_rounds': 10, 'n_hill_climb_restarts': 10, 'random_walk_prob': 0.100, 'initial_set_noise_scale': 0.050},
    ],
}



ensemble_base_model_configs = [
    # Top performer: THCR
    (THCR, {'smoothing': 0.050, 'explore_rounds': 10}),
    
    # Strong performer: AdaptiveBudgetCombinatorialBandit
    (AdaptiveBudgetCombinatorialBandit, {'alpha': 0.200, 'initial_q_value': 0.500, 'initial_counts': 1}),
    
    # Good performer: GreedyCostEfficiency - Using a small explore_rounds to allow some initial exploration
    # and a tiny noise scale for potential tie-breaking/slight randomization.
    (GreedyCostEfficiency, {'explore_rounds': 0, 'initial_q_value': 1.000, 'exploitation_noise_scale': 0.000, 'initial_counts': 1}),

    # Solid performer: UCB_MB
    (UCB_MB, {'c': 0.050, 'initial_q_value': 1.000, 'initial_counts': 1}),

    # Including standard UCB, which also performed reasonably well
    (UCB, {'c': 0.005, 'initial_q_value': 1.0, 'initial_counts': 1}), # Using initial_q_value/counts for consistency
    
    # Including ThompsonSampling, though slightly lower, for diversity in exploration strategy
    (ThompsonSampling, {'alpha_prior': 0.050, 'beta_prior': 1.000}),
]

def get_base_agent_configurations():
    configs = base_configs
    agents = [
        # These are good
        # THCR,
        # UCB,
        # ThompsonSampling,
        # UCB_MB,
        # BudgetedThompsonSampling,
        # GreedyCostEfficiency,
        # AdaptiveBudgetCombinatorialBandit,
        # FractionalKnapsackDecreasingEpsilonGreedy,

        # CBwKGreedyUCB,
        CBwKUCBV,
        # CBwKTunedUCB,

        
        # These are not good
        # EpsilonFirstBudget,
        # EpsilonGreedy,
        # GreedyBudget,
        # SoftmaxBudget,
        # ExploreThenCommitBudget,
        # LinUCB,
        # ThompsonSamplingSimilarity,
        # MF_UCB,

        # This is shit
        # EnsembleWeightedBandit,
    ]
    return {agent: configs[agent] for agent in agents}