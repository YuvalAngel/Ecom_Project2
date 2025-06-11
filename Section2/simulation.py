import numpy as np

def simulate(AgentClass, test, **kwargs):
    """
    Simulate the recommendation process over a series of weeks for a given agent and test scenario.

    Parameters:
    - AgentClass: class of the recommendation agent (e.g., Recommender or bandit agent)
    - test: dictionary containing test configuration (P, budget, prices, n_weeks, etc.)
    - kwargs: additional parameters to initialize the agent

    Returns:
    - cumulative_reward: total sum of positive feedback (rewards) over all weeks and users
    """
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
    """
    Format parameters dictionary into a string-friendly representation.

    Floats are formatted to 2 decimal places.

    Parameters:
    - params: dict of parameter names and values

    Returns:
    - dict with formatted string values
    """
    return {k: (f"{v:.3f}" if isinstance(v, (float, np.floating)) else v) for k, v in params.items()}


def filter_within_k_percent(top_configs, k_percent):
    """
    Filter configurations that have rewards within 10% of the top reward for each agent.

    Parameters:
    - top_configs: dict mapping AgentClass to list of (params, reward) tuples sorted descending
    - k_percent: float percentage to filter by

    Returns:
    - filtered: dict with same keys but only configs within 90% of top reward kept
    """
    filtered = {}
    for AgentClass, configs in top_configs.items():
        if not configs:
            filtered[AgentClass] = []
            continue
        top_reward = configs[0][1]
        threshold = top_reward * k_percent
        filtered_configs = [(params, reward) for params, reward in configs if reward >= threshold]
        filtered[AgentClass] = filtered_configs
    return filtered