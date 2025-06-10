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


import numpy as np

def base_configurations():
    return {
        Recommender: [
            # For smoothing: expand around 0.10, 0.50, 1.75 with fine granularity
            *[
                {'smoothing': s, 'explore_rounds': er, 'max_iters': mi}
                for s in np.linspace(0.05, 0.2, 10)  # around 0.10
                for er, mi in [(10, 50)]
            ],
            *[
                {'smoothing': s, 'explore_rounds': er, 'max_iters': mi}
                for s in np.linspace(0.3, 0.7, 10)  # around 0.50
                for er, mi in [(15, 70)]
            ],
            *[
                {'smoothing': s, 'explore_rounds': er, 'max_iters': mi}
                for s in np.linspace(1.5, 2.0, 10)  # around 1.75
                for er, mi in [(20, 100)]
            ],
        ],

        EpsilonGreedy: [
            {'epsilon': e} for e in np.linspace(0.10, 0.20, 30)  # centered around 0.13–0.17
        ],

        UCB: [
            {'c': c} for c in np.linspace(0.05, 0.35, 30)  # around 0.10–0.30
        ],

        ThompsonSampling: [
            {'alpha': a} for a in np.linspace(0.001, 0.35, 30)  # around 0.01–0.30 with a bit more
        ],
    }


def filter_within_10_percent(top_configs):
    filtered = {}
    for AgentClass, configs in top_configs.items():
        if not configs:
            filtered[AgentClass] = []
            continue
        top_reward = configs[0][1]  # assuming configs sorted descending by reward
        threshold = top_reward * 0.9
        filtered_configs = [(params, reward) for params, reward in configs if reward >= threshold]
        filtered[AgentClass] = filtered_configs
    return filtered



def run_tests(num_runs_per_test=1, top_k=5, agents_params=None):
    total_rewards = {}  # keys: (AgentClass, frozenset(params.items())), values: cumulative reward
    
    if agents_params is None:
        agents_params = base_configurations()
    
    for i, test in enumerate(tests):
        print(f"\n=== Test {i+1} (Expected ≥ {required_results[i]}) ===")
        
        for AgentClass, param_list in agents_params.items():
            best_reward = -np.inf
            best_params = None
            
            for params in tqdm(param_list, desc=f"{AgentClass.__name__} Params", leave=False):
                rewards = []
                for _ in range(num_runs_per_test):
                    reward = simulate(AgentClass, test, **params)
                    rewards.append(reward)
                avg_reward = np.mean(rewards)
                
                # Accumulate total rewards across tests and runs
                key = (AgentClass, frozenset(params.items()))
                total_rewards[key] = total_rewards.get(key, 0) + avg_reward
                
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    best_params = params
            
            status = "✅" if best_reward >= required_results[i] else "❌"
            print(f"{AgentClass.__name__:>16}: Best Reward = {best_reward:.2f} {status} Using Config: {format_params(best_params)}")
    
    # Sort total_rewards and get top_k configs per agent
    # total_rewards keys: (AgentClass, frozenset(params.items()))
    by_agent = {}
    for (AgentClass, params_frozen), total_reward in total_rewards.items():
        by_agent.setdefault(AgentClass, []).append((dict(params_frozen), total_reward))
    
    top_configs = {}
    for AgentClass, results in by_agent.items():
        # Sort descending by total_reward and pick top_k
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
        top_configs[AgentClass] = sorted_results
    
    return top_configs


def format_top_configs(top_configs, label):
    print(f"\n=== Top Configurations ({label}) ===")
    for AgentClass, configs in top_configs.items():
        print(f"\n{AgentClass.__name__}:")
        for i, (params, total_reward) in enumerate(configs, 1):
            print(f"  {i}. Reward = {total_reward:.2f} with params {format_params(params)}")


if __name__ == "__main__":
    print("Starting Initial Run: 1 run per configuration")
    quick_top = run_tests(num_runs_per_test=1)
    format_top_configs(quick_top, "Quick Run (1x per test)")

    # Filter configs within 10% of top reward
    quick_filtered = filter_within_10_percent(quick_top)

    # Extract params for next stage
    middle_agents_params = {
        AgentClass: [params for params, _ in configs]
        for AgentClass, configs in quick_filtered.items()
    }

    print("Starting Middle Run: 10 runs per configuration")
    middle_top = run_tests(num_runs_per_test=10, agents_params=middle_agents_params)
    format_top_configs(middle_top, "Middle Run (10x per test)")

    middle_filtered = filter_within_10_percent(middle_top)
    final_agents_params = {
        AgentClass: [params for params, _ in configs]
        for AgentClass, configs in middle_filtered.items()
    }

    print("Starting Final Run: 50 runs per configuration")
    final_top = run_tests(num_runs_per_test=50, agents_params=final_agents_params)
    format_top_configs(final_top, "Final Run (50x per test)")

    print("\n=== 🎉 Final Best Configurations Within 10% of Top Reward 🎉 ===")
    for AgentClass, configs in final_top.items():
        top_reward = configs[0][1] if configs else 0
        threshold = top_reward * 0.9
        filtered_configs = [(params, reward) for params, reward in configs if reward >= threshold]
        print(f"\n{AgentClass.__name__}:")
        for i, (params, total_reward) in enumerate(filtered_configs, 1):
            print(f"  {i}. Total Reward = {total_reward:.2f} with params {format_params(params)}")
