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

    THCR: [
        # Baseline configurations (similar to your previous ones, but with new hill-climbing params set to a starting point)
        {'smoothing': 0.05, 'explore_rounds': 10, 'n_hill_climb_restarts': 5, 'random_walk_prob': 0.05, 'initial_set_noise_scale': 0.01},
        {'smoothing': 0.05, 'explore_rounds': 20, 'n_hill_climb_restarts': 5, 'random_walk_prob': 0.05, 'initial_set_noise_scale': 0.01},

        # Tuning random_walk_prob (higher chance of escaping local optima)
        {'smoothing': 0.05, 'explore_rounds': 10, 'n_hill_climb_restarts': 5, 'random_walk_prob': 0.10, 'initial_set_noise_scale': 0.01},
        {'smoothing': 0.05, 'explore_rounds': 10, 'n_hill_climb_restarts': 5, 'random_walk_prob': 0.15, 'initial_set_noise_scale': 0.01},

        # Tuning initial_set_noise_scale (more varied starting points for restarts)
        {'smoothing': 0.05, 'explore_rounds': 10, 'n_hill_climb_restarts': 5, 'random_walk_prob': 0.05, 'initial_set_noise_scale': 0.05},
    ],

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

hashable_ensemble_base_model_configs = tuple([
    (model_class, frozenset(config.items()))
    for model_class, config in ensemble_base_model_configs
])


base_configs[EnsembleWeightedBandit] = [
    {
        'base_models_and_configs': hashable_ensemble_base_model_configs,
        'learning_rate': 0.01, # Lower learning rate for more stable weight changes
        'initial_weight_value': 1.0, # Start with equal weights
    },
    {
        'base_models_and_configs': hashable_ensemble_base_model_configs,
        'learning_rate': 0.05, # Moderate learning rate
        'initial_weight_value': 1.0,
    },
    {
        'base_models_and_configs': hashable_ensemble_base_model_configs,
        'learning_rate': 0.1, # Your original learning rate, good for quicker adaptation
        'initial_weight_value': 1.0,
    },
    {
        'base_models_and_configs': hashable_ensemble_base_model_configs,
        'learning_rate': 0.2, # Higher learning rate for more aggressive adaptation
        'initial_weight_value': 1.0,
    },
    # You might also consider experimenting with `initial_weight_value`
    # e.g., giving higher initial weights to models you expect to perform better
    # based on prior runs.
]


def get_base_agent_configurations():
    configs = base_configs
    agents = [
        # These are good
        THCR,
        # UCB,
        # ThompsonSampling,
        # UCB_MB,
        # BudgetedThompsonSampling,
        # GreedyCostEfficiency,
        # AdaptiveBudgetCombinatorialBandit,
        # FractionalKnapsackDecreasingEpsilonGreedy,

        
        # These are not good
        # EpsilonFirstBudget,
        # EpsilonGreedy,
        # GreedyBudget,
        # SoftmaxBudget,
        # ExploreThenCommitBudget,
        # LinUCB,
        # ThompsonSamplingSimilarity,
        # MF_UCB,
        # EnsembleWeightedBandit,
    ]
    return {agent: configs[agent] for agent in agents}