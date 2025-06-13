from runner import single_run
from config import *


if __name__ == "__main__":
    configs = get_base_agent_configurations()
    single_run(configs, iterations=10)
