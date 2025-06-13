import numpy as np
from tqdm import tqdm
from simulation import Simulation
from test import tests, required_results


def format_params(params):
    """
    Format parameters dictionary into a string-friendly representation.
    Floats are formatted to 2 decimal places.
    Parameters:
    - params: dict of parameter names and values
    Returns:
    - dict with formatted string values
    """
    return ", ".join([f"{k}={v:.3f}" if isinstance(v, (float, np.floating)) else f"{k}={v}" for k, v in params.items()])


def filter_within_range(top_configs, range_val=100): # Renamed 'range' to 'range_val' to avoid conflict with built-in range
    """
    Filter configurations that have rewards within value of the top reward for each agent.
    Parameters:
    - top_configs: dict mapping AgentClass to list of (params, reward) tuples sorted descending
    - range_val: int range of top reward we want to filter 
    Returns:
    - filtered: dict with same keys but only configs within the specified range of top reward kept
    """
    filtered = {}
    for AgentClass, configs in top_configs.items():
        if not configs:
            filtered[AgentClass] = []
            continue
        top_reward = configs[0][1]
        threshold = top_reward - range_val
        filtered_configs = [(params, reward) for params, reward in configs if reward >= threshold]
        filtered[AgentClass] = filtered_configs
    return filtered


def run_tests(agents_params, num_iterations=1, top_k=5):
    """
    Run all tests for each configuration for a specified number of iterations.
    Each iteration involves running all defined tests.
    The total reward for an iteration is the sum of rewards from all tests in that iteration.
    The average reward is then computed over these iteration sums.
    Also provides per-test feedback against required results, ordered best to worst.
    """
    all_iterations_total_rewards = {} 
    per_test_rewards_tracker = {}

    for AgentClass, param_list in agents_params.items():
        for params in param_list:
            key = (AgentClass, frozenset(params.items()))
            all_iterations_total_rewards[key] = [] 
            per_test_rewards_tracker[key] = {i: [] for i in range(len(tests))}

            for iteration_num in tqdm(range(num_iterations), 
                                      desc=f"Iterations for {AgentClass.__name__} {format_params(params)}", 
                                      leave=True):
                
                current_iteration_total_reward = 0.0
                
                for i, test in enumerate(tests):
                    sim_instance = Simulation(
                        P=test['P'],
                        prices=test['item_prices'],
                        budget=test['budget'],
                        n_weeks=test['n_weeks'],
                        agent_class=AgentClass,
                        agent_params=params
                    )
                    reward_from_test_run = sim_instance.run()
                    current_iteration_total_reward += reward_from_test_run
                    per_test_rewards_tracker[key][i].append(reward_from_test_run)
                
                all_iterations_total_rewards[key].append(current_iteration_total_reward)

    # --- Start of the CORRECTED printing section (ordered) ---
    print("\n=== Per-Test Performance ===")
    
    # Loop through each test globally first to group results by test
    for i, test in enumerate(tests):
        print(f"\n--- Test {i+1} (Expected ≥ {required_results[i]}) ---")
        
        # Collect results for this specific test across all configurations
        test_results_for_printing = []

        for key in per_test_rewards_tracker.keys():
            AgentClass, params_frozen = key
            params = dict(params_frozen) 

            rewards_for_this_test_config = per_test_rewards_tracker[key][i]
            avg_reward_for_this_test_config = np.mean(rewards_for_this_test_config) if rewards_for_this_test_config else 0.0

            status = "✅" if avg_reward_for_this_test_config >= required_results[i] else "❌"
            
            test_results_for_printing.append({
                'AgentClass': AgentClass,
                'params': params,
                'avg_reward': avg_reward_for_this_test_config,
                'status': status
            })
        
        # Sort the collected results for the current test from best to worst average reward
        test_results_for_printing.sort(key=lambda x: x['avg_reward'], reverse=True)

        # Print the sorted results
        best_reward_for_this_test = -np.inf # Re-initialize to find the best among the sorted list
        best_params_for_this_test = None
        best_agent_class_for_this_test = None

        for result in test_results_for_printing:
            print(f"{result['AgentClass'].__name__:>16} with {format_params(result['params'])}: Avg Reward for Test {i+1} = {result['avg_reward']:.2f} {result['status']}")
            
            # Keep track of the absolute best for the summary line
            if result['avg_reward'] > best_reward_for_this_test:
                best_reward_for_this_test = result['avg_reward']
                best_params_for_this_test = result['params']
                best_agent_class_for_this_test = result['AgentClass']
        
        if best_agent_class_for_this_test:
            print(f"   Best for Test {i+1}: {best_agent_class_for_this_test.__name__} with {format_params(best_params_for_this_test)} (Reward: {best_reward_for_this_test:.2f})")
    # --- End of the CORRECTED printing section ---

    # Calculate final overall average rewards (sum of test rewards, averaged over iterations)
    final_avg_rewards = {}
    for (AgentClass, params_frozen), rewards_list in all_iterations_total_rewards.items():
        if rewards_list:
            final_avg_rewards[(AgentClass, params_frozen)] = np.mean(rewards_list)
        else:
            final_avg_rewards[(AgentClass, params_frozen)] = 0.0 

    # Prepare top_configs structure for output
    by_agent = {}
    for (AgentClass, params_frozen), avg_total_reward in final_avg_rewards.items():
        by_agent.setdefault(AgentClass, []).append((dict(params_frozen), avg_total_reward))

    top_configs = {}
    for AgentClass, results in by_agent.items():
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
        top_configs[AgentClass] = sorted_results

    return top_configs


def format_top_configs(top_configs, label):
    """
    Nicely print the top configurations per agent.
    """
    print(f"\n=== Top Configurations ({label}) ===")
    for AgentClass, configs in top_configs.items():
        print(f"\n{AgentClass.__name__}:")
        for i, (params, overall_avg_reward) in enumerate(configs, 1):
            print(f"  {i}. Overall Average Reward = {overall_avg_reward:.2f} with params {format_params(params)}")

def single_run(configs, iterations=50, range_val=100):
    """
    Single-stage run evaluating all base configurations with a specified number of iterations.
    """
    print(f"\nStarting Run: {iterations} full test iterations per configuration")
    best_configs = run_tests(configs, num_iterations=iterations) 
    
    filtered_configs = filter_within_range(best_configs, range_val)

    format_top_configs(filtered_configs, f"{iterations} iterations per config")