import numpy as np
from tqdm import tqdm
from simulation import *
from test import *  # import your tests and requirements

def run_tests(agents_params, num_runs_per_test=1, top_k=5):
    """
    Run all tests with specified number of repetitions per configuration, and collect results.

    For each agent and each configuration:
    - Run the simulation num_runs_per_test times
    - Average the results
    - Track best performing configuration per test
    - Aggregate total rewards across all tests for each configuration

    Parameters:
    - agents_params: dict mapping AgentClass to list of parameter dicts to test
    - num_runs_per_test: int, how many times to repeat each config per test for averaging
    - top_k: int, number of top configurations to keep per agent after all tests

    Returns:
    - top_configs: dict mapping AgentClass to list of top_k (params, total_reward) tuples sorted descending
    """
    total_rewards = {}

    for i, test in enumerate(tests):
        print(f"\n=== Test {i+1} (Expected ≥ {required_results[i]}) ===")

        for AgentClass, param_list in agents_params.items():
            best_reward = -np.inf
            best_params = None

            for params in param_list:
                rewards = [simulate(AgentClass, test, **params)
                           for _ in tqdm(range(num_runs_per_test), 
                             desc=f"{AgentClass.__name__} - {format_params(params)}", 
                             leave=True)]
                avg_reward = np.mean(rewards)

                key = (AgentClass, frozenset(params.items()))
                total_rewards[key] = total_rewards.get(key, 0) + avg_reward

                if avg_reward > best_reward:
                    best_reward = avg_reward
                    best_params = params

            status = "✅" if best_reward >= required_results[i] else "❌"
            print(f"{AgentClass.__name__:>16}: Best Average Reward over Iterations = {best_reward:.2f} {status} Using Config: {format_params(best_params)}")

    by_agent = {}
    for (AgentClass, params_frozen), total_reward in total_rewards.items():
        by_agent.setdefault(AgentClass, []).append((dict(params_frozen), total_reward))

    top_configs = {}
    for AgentClass, results in by_agent.items():
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
        top_configs[AgentClass] = sorted_results

    return top_configs


def format_top_configs(top_configs, label):
    """
    Nicely print the top configurations per agent.

    Parameters:
    - top_configs: dict mapping AgentClass to list of (params, total_reward) tuples
    - label: str, label to describe the run (e.g., "Final Run")
    """
    print(f"\n=== Top Configurations ({label}) ===")
    for AgentClass, configs in top_configs.items():
        print(f"\n{AgentClass.__name__}:")
        for i, (params, total_reward) in enumerate(configs, 1):
            print(f"  {i}. Best Avg Reward = {total_reward:.2f} with params {format_params(params)}")



def multi_run(configs, first_run=3, second_run=10, third_run=25, range=100):
    """
    Multi-stage run to refine best configurations.

    Stages:
    1) Quick run with few repetitions (first_run)
    2) Middle run with more repetitions on filtered top configs (second_run)
    3) Final run with many repetitions on filtered configs from middle stage (third_run)

    Each stage filters configs within 10% of the top reward to pass to next stage.

    Parameters:
    - configs: dict, mapping AgentClass to configuration
    - first_run: int, number of runs per config in initial quick run
    - second_run: int, runs per config in middle run
    - third_run: int, runs per config in final run
    """
    print(f"\nStarting Initial Run: {first_run} runs per configuration")

    quick_top = run_tests(configs, num_runs_per_test=first_run)
    format_top_configs(quick_top, f"Quick Run ({first_run}x per test)")
    quick_filtered = filter_within_range(quick_top, range)



    middle_agents_params = {AgentClass: [params for params, _ in configs] for AgentClass, configs in quick_filtered.items()}
    print(f"\nStarting Middle Run: {second_run} runs per configuration")
    middle_top = run_tests(middle_agents_params, num_runs_per_test=second_run)
    middle_filtered = filter_within_range(middle_top, range)
    format_top_configs(middle_filtered, f"Middle Run ({second_run}x per test)")

    final_agents_params = {AgentClass: [params for params, _ in configs] for AgentClass, configs in middle_filtered.items()}
    print(f"\nStarting Final Run: {third_run} runs per configuration")
    final_top = run_tests(final_agents_params, num_runs_per_test=third_run)
    final_filtered = filter_within_range(final_top, range)
    print(f"\n=== 🎉 Final Best Configurations Within {range} Points of Top Reward 🎉 ===")
    format_top_configs(final_filtered, f"Final Run ({third_run}x per test)")



def single_run(configs, iterations=50, range=100):
    """
    Single-stage run evaluating all base configurations with a specified number of repetitions.

    Parameters:
    - configs: dict, mapping AgentClass to configuration
    - iterations: int, number of runs per configuration per test
    """
    print(f"\nStarting Run: {iterations} runs per configuration")
    best_configs = run_tests(configs, num_runs_per_test=iterations)
    
    # 🧹 Filter configs within 10% of top reward
    filtered_configs = filter_within_range(best_configs, range)

    # 🖨️ Print formatted top configs
    # format_top_configs(filtered_configs, f"{iterations}x per test (Filtered)")

    # print(f"\n=== 🎉 Final Best Configurations Within {range} Points of Top Reward 🎉 ===")
    format_top_configs(filtered_configs, f"{iterations}x per test")