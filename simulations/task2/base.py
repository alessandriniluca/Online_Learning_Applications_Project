import numpy as np

from common.utils import load_static_env_configuration, load_static_sim_configuration, get_test_alphas_functions
from environment.environment import Environment
from optimizer.estimator import Estimator
from optimizer.optimizer import Optimizer

np.set_printoptions(formatter={'float': lambda x: "{0:0.10f}".format(x)})


env_configuration = load_static_env_configuration("../../configurations/environment/static_conf_1.json")
sim_configuration = load_static_sim_configuration("../../configurations/simulation/sim_conf_1.json")
alphas_functions = get_test_alphas_functions()

env = Environment(
    configuration=env_configuration,
    alphas_functions=alphas_functions
)

estimator = Estimator(env.configuration.graph_clicks,
                      env.products,
                      env.configuration.lambda_prob,
                      env.configuration.reservation_price_means,
                      env.configuration.reservation_price_std_dev
                      )

buy_probs = estimator.get_buy_probs()

optimizer = Optimizer(
    users_number=env.configuration.average_users_number,
    min_budget=sim_configuration["min_budget"],
    max_budget=sim_configuration["max_budget"],
    total_budget=sim_configuration["total_budget"],
    resolution=sim_configuration["resolution"],
    products=env.products,
    mean_quantities=env.configuration.quantity_means,
    buy_probs=buy_probs,
    basic_alphas=env.configuration.basic_alphas,
    alphas_functions=alphas_functions
)

# Optimize 15 campaigns
optimizer.run_optimization()
best_allocation = optimizer.find_best_allocation()
print(best_allocation)

# Optimize 5 campaigns
optimizer.one_campaign_per_product = True
optimizer.run_optimization()
best_allocation = optimizer.find_best_allocation()
print(best_allocation)


if __name__ == '__main__':
    print("simulation done")
