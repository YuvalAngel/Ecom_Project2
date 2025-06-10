from runner import single_run, multi_run
from config import *


if __name__ == "__main__":
    configs = get_base_agent_configurations()
    single_run(configs, iterations=20)
    # multi_run(configs)
