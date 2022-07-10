import numpy as np

from common.utils import load_static_configuration
from environment.environment import Environment
from optimizer.optimizer import Optimizer
from optimizer.optimizer2 import Optimizer2
from probability_calculator.probabilities import Probabilities

configuration = load_static_configuration("configurations/static_conf_1.json")

test_env = Environment(
    configuration=configuration
)

test_env.round([0, 0, 0, 0, 0])

if __name__ == '__main__':
    print("ok")
