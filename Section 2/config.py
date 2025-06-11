from recommenders import *


base_configs = {
        THCR: [
            {'smoothing': 0.10, 'explore_rounds': 20},
        ],

        UCB: [
            # {'c': 0.001},
            {'c': 0.01},
            # {'c': 0.05},
            # {'c': 0.10},
            # {'c': 0.15},
        ],

        ThompsonSampling: [
            # {'alpha': 0.01},
            # {'alpha': 0.03},
            # {'alpha': 0.05},
            {'alpha_prior': 0.1, 'beta_prior': 1.0},
            # {'alpha_prior': 0.2, 'beta_prior': 5.0},
            # {'alpha_prior': 1.0, 'beta_prior': 8.0},
            # {'alpha_prior': 0.2, 'beta_prior': 2.0},

        ],
        EpsilonGreedy: [
            {'epsilon': 0.10, 'decay': 0.99},
            {'epsilon': 0.30, 'decay': 0.95},
            {'epsilon': 0.50, 'decay': 0.95},
        ],
    }



def get_base_agent_configurations():
    
    configs = base_configs


    # Comment in agents you wish to run
    agents = [
        # THCR,
        # EpsilonGreedy,
        UCB,
        # ThompsonSampling,
    ]

    return {agent: configs[agent] for agent in agents}