from recommender import *


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
    }