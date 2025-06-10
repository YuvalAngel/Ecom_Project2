import numpy as np
from recommender import Recommender, EpsilonGreedy, UCB, ThompsonSampling
import matplotlib.pyplot as plt
from test import tests, required_results
from tqdm import tqdm


def simulate(AgentClass, test, **kwargs):
    P = test['P']
    N, K = P.shape
    B = test['budget']
    prices = test['item_prices']
    T = test['n_weeks']

    if AgentClass.__name__ == "Recommender":
        agent = AgentClass(n_weeks=T, n_users=N, prices=prices, budget=B, **kwargs)
    else:
        agent = AgentClass(n_users=N, n_arms=K, prices=prices, budget=B, **kwargs)

    cumulative_reward = 0

    for _ in range(T):
        recs = agent.recommend()
        probs = np.array([P[u, a] for u, a in enumerate(recs)])
        feedback = (np.random.rand(N) < probs).astype(int)
        cumulative_reward += feedback.sum()

        if AgentClass.__name__ == "Recommender":
            agent.update(feedback)
        else:
            agent.update(users=np.arange(N), arms=recs, rewards=feedback)

    return cumulative_reward


def format_params(params):
    return {k: (f"{v:.2f}" if isinstance(v, (float, np.floating)) else v) for k, v in params.items()}


def run_tests():
    total_rewards = {}  # keys: (AgentClass, frozenset(params.items())), values: cumulative reward
    agents_params = {
    Recommender: [
        {'smoothing': 0.1, 'explore_rounds': 10, 'max_iters': 50},
        {'smoothing': 1.0, 'explore_rounds': 5, 'max_iters': 50},
        {'smoothing': 1.75, 'explore_rounds': 20, 'max_iters': 100},
        {'smoothing': 0.5, 'explore_rounds': 15, 'max_iters': 70},
        {'smoothing': 2.0, 'explore_rounds': 25, 'max_iters': 100},
    ],
    EpsilonGreedy: [
        {'epsilon': e} for e in np.linspace(0.01, 0.3, 30)
    ],
    UCB: [
        {'c': c} for c in np.linspace(0.1, 3.0, 30)
    ],
    ThompsonSampling: [
        {'alpha': a} for a in np.linspace(0.001, 0.3, 30)
    ]
}

    for i, test in enumerate(tests):
        print(f"\n=== Test {i+1} (Expected ≥ {required_results[i]}) ===")
        for AgentClass, param_list in agents_params.items():
            best_reward = -np.inf
            best_params = None
            for params in tqdm(param_list, desc=f"{AgentClass.__name__} Params", leave=False):
                reward = simulate(AgentClass, test, **params)

                # Always add reward for this config (accumulate over tests)
                key = (AgentClass, frozenset(params.items()))
                total_rewards[key] = total_rewards.get(key, 0) + reward

                if reward > best_reward:
                    best_reward = reward
                    best_params = params

            status = "✅" if best_reward >= required_results[i] else "❌"
            print(f"{AgentClass.__name__:>16}: reward = {best_reward} {status} with params {format_params(best_params)}")
    
    best_combo, best_total_reward = max(total_rewards.items(), key=lambda x: x[1])
    best_agent_class, best_params_frozen = best_combo
    best_params = dict(best_params_frozen)

    print("\n=== 🎉 Best Overall Configuration 🎉 ===")
    print(f"🤖 Agent: {best_agent_class.__name__}")
    print(f"⚙️ Params: {format_params(best_params)}")
    print(f"🏆 Total Reward Across All Tests: {best_total_reward}")



if __name__ == "__main__":
    run_tests()