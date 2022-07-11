import numpy as np

from common.utils import load_static_env_configuration, load_static_sim_configuration
from environment.environment import Environment
from optimizer.estimator import Estimator
from optimizer.optimizer import Optimizer

np.set_printoptions(formatter={'float': lambda x: "{0:0.10f}".format(x)})


env_configuration = load_static_env_configuration("../../configurations/environment/static_conf_1.json")
sim_configuration = load_static_sim_configuration("../../configurations/simulation/sim_conf_1.json")

env = Environment(
    configuration=env_configuration
)

estimator = Estimator(env.configuration.graph_clicks,
                      env.products,
                      env.configuration.lambda_prob,
                      env.configuration.reservation_price_means,
                      env.configuration.reservation_price_std_dev
                      )

buy_probs = estimator.get_buy_probs()

products = env.products
quantities = env.configuration.quantity_means

min_budget = sim_configuration["min_budget"]
max_budget = sim_configuration["max_budget"]
total_budget = sim_configuration["total_budget"]
resolution = sim_configuration["resolution"]

prices = []
for product in products:
    prices.append(product.price)

# Compute all alpha primes: the new alpha ratio that I will have if a budget was allocated
# Note that we will get expected value of dirichlet variables that are used to sample alphas
alphas_prime = np.zeros((int(total_budget / resolution) + 1, len(products), 3))

# for each budget allocation
for single_budget in range(0, total_budget + resolution, resolution):
    # for each product
    for product_index in range(len(products)):
        # for each class of user
        for class_index, users_of_current_class in enumerate(env.configuration.average_users_number):
            # set budget to corresponding product (using array with zeros for alpha function compatibility)
            # allocate just for one product the budget
            budgets = np.zeros(5)
            budgets[product_index] = single_budget
            # compute deltas of weights
            delta_alpha_weights = env.alphas_functions[class_index](budgets)
            # concatenate a zero in order to consider also lost visitors
            delta_alpha_weights = np.concatenate((delta_alpha_weights, np.array([0])))
            # compute alpha primes (expected value of dirichlet variables)
            alphas_prime[int(single_budget / resolution)][product_index][class_index] = \
                (env.configuration.basic_alphas[class_index] + delta_alpha_weights)[product_index] / sum(
                (env.configuration.basic_alphas[class_index]))

print(alphas_prime)

delta_alpha_weights = env.alphas_functions[0]()

optimizer = Optimizer(
    users_number=env.configuration.average_users_number,
    min_budget=min_budget,
    max_budget=max_budget,
    total_budget=total_budget,
    resolution=resolution,
    prices=prices,
    mean_quantities=env.configuration.quantity_means,
    alphas=alphas_prime,
    buy_probs=buy_probs
)

# Optimize 5 campaigns
optimizer.run_optimization()
best_allocation = optimizer.find_best_allocation()
print(best_allocation)

# Optimize 5 campaigns (exploit contexts)
optimizer.set_exploit_context(True)
optimizer.run_optimization()
best_allocation = optimizer.find_best_allocation()
print(best_allocation)

if __name__ == '__main__':
    print("simulation done")
