import numpy as np

from common.utils import load_static_env_configuration, get_test_alphas_functions
from environment.environment_context import Environment

np.set_printoptions(formatter={'float': lambda x: "{0:0.10f}".format(x)})

configuration = load_static_env_configuration("../configurations/environment/static_conf_1.json")
alphas_functions = get_test_alphas_functions()

test_env = Environment(
    configuration=configuration,
    alphas_functions=alphas_functions
)

test_env.round([100, 10, 10, 100, 10, 10, 10, 0, 0, 0], [[0, 1], [2, 3]])

print(test_env.alphas_functions[0]([100,0,0,0,0]))

# new_weights = np.array([10, 20, 10, 10, 10, 40])
# samples = np.zeros(6)
# for i in range(1000):
#     samples += (np.random.dirichlet(new_weights / sum(new_weights)))
#
# print(samples / 1000)



if __name__ == '__main__':
    print("ok")
