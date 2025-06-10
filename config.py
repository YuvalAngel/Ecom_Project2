from recommenders import *


base_configs = {
        Recommender: [
            {'max_iters': 50, 'smoothing': 0.10, 'explore_rounds': 10},
            {'max_iters': 50, 'smoothing': 0.15, 'explore_rounds': 10},
            {'max_iters': 50, 'smoothing': 0.11, 'explore_rounds': 10},
            {'explore_rounds': 15, 'max_iters': 70, 'smoothing': 0.38},
            {'max_iters': 50, 'smoothing': 0.50, 'explore_rounds': 10},
        ],

        EpsilonGreedy: [
            {'epsilon': 0.10},
            {'epsilon': 0.17},
            {'epsilon': 0.14},
            {'epsilon': 0.12},
            {'epsilon': 0.11},
        ],

        UCB: [
            {'c': 0.21},
            {'c': 0.13},
            {'c': 0.14},
            {'c': 0.29},
            {'c': 0.10},
        ],

        ThompsonSampling: [
            {'alpha': 0.01},
            {'alpha': 0.17},
            {'alpha': 0.11},
            {'alpha': 0.24},
            {'alpha': 0.27},
        ],
        EpsilonGreedyImproved: [
            {'epsilon': 0.30, 'decay': 0.95},
            {'epsilon': 0.50, 'decay': 0.95},
            {'epsilon': 0.30, 'decay': 0.97},
            {'epsilon': 0.50, 'decay': 0.97},
        ],
    }



def get_base_agent_configurations():
    
    configs = base_configs


    # Comment in agents you wish to run
    agents = [
        # Recommender,
        # EpsilonGreedy,
        UCB,
        # ThompsonSampling,
        # EpsilonGreedyImproved
    ]

    return {agent: configs[agent] for agent in agents}