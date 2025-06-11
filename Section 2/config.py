from recommenders import *


base_configs = {
        Recommender: [
            {'max_iters': 30, 'smoothing': 0.05, 'explore_rounds': 10},
            {'max_iters': 50, 'smoothing': 0.10, 'explore_rounds': 10},
            {'max_iters': 50, 'smoothing': 0.11, 'explore_rounds': 10},
            {'max_iters': 50, 'smoothing': 0.05, 'explore_rounds': 20},
            {'max_iters': 50, 'smoothing': 0.10, 'explore_rounds': 20},
            # {'max_iters': 70, 'smoothing': 0.38, 'explore_rounds': 15},
        ],

        UCB: [
            # {'c': 0.10},
            {'c': 0.13},
            {'c': 0.14},
            {'c': 0.20},
            {'c': 0.21},
        ],

        ThompsonSampling: [
            {'alpha': 0.01},
            {'alpha': 0.03},
            {'alpha': 0.05},
            # {'alpha': 0.15},
            # {'alpha': 0.17},
            # {'alpha': 0.20},
        ],
        EpsilonGreedy: [
            {'epsilon': 0.10, 'decay': 0.995},
            {'epsilon': 0.30, 'decay': 0.995},
            {'epsilon': 0.50, 'decay': 0.995},
            {'epsilon': 0.10, 'decay': 0.99},
            {'epsilon': 0.30, 'decay': 0.99},
            {'epsilon': 0.50, 'decay': 0.99},
            # {'epsilon': 0.30, 'decay': 0.97},
            # {'epsilon': 0.50, 'decay': 0.97},
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
    ]

    return {agent: configs[agent] for agent in agents}