import numpy as np

from common.utils import load_static_configuration
from environment.environment import Environment
from optimizer.optimizer import Optimizer
from optimizer.optimizer2 import Optimizer2
from probability_calculator.probabilities import Probabilities


np.set_printoptions(formatter={'float': lambda x: "{0:0.10f}".format(x)})

configuration = load_static_configuration("configurations/static_conf_1.json")
print(configuration)

env = Environment(
    configuration=configuration
)

calculator = Probabilities(env.configuration.graph_clicks,
                           env.products,
                           env.configuration.lambda_prob,
                           env.configuration.reservation_price_means,
                           env.configuration.reservation_price_std_dev
                           )
buy_probs = calculator.get_buy_probs()


products = env.products
quantities = env.configuration.quantity_means

total_budget = 180
resolution = 10
min_budget = [0, 0, 0, 0, 0]
max_budget = [100, 100, 100, 100, 100]

prices = []
for product in products:
    prices.append(product.price)

alphas_prime = np.zeros((int(total_budget / resolution) + 1, len(products), 3))

for single_budget in range(0, total_budget + resolution, resolution):
    for product_index in range(len(products)):
        for i, users in enumerate(env.configuration.average_users_number):
            # set budget to corresponding product (using array with zeros for alpha function compatibility)
            budgets = np.zeros(5)
            budgets[product_index] = single_budget
            # compute deltas of weights
            delta_alpha_weights = env.alphas_functions[i](budgets)
            # concatenate a zero in order to consider also lost visitors
            delta_alpha_weights = np.concatenate((delta_alpha_weights, np.array([0])))
            # compute alpha primes (expected value of dirichlet variables)
            alphas_prime[int(single_budget / resolution)][product_index][i] = \
                (env.configuration.basic_alphas[i] + delta_alpha_weights)[product_index] / sum(
                (env.configuration.basic_alphas[i] + delta_alpha_weights))
print("EEEEEEE", alphas_prime, "FFFFFFFF")

optimizer = Optimizer(
    env.configuration.average_users_number,
    min_budget,
    max_budget,
    buy_probs,
    prices,
    total_budget,
    resolution,
    quantities)
optimizer.set_alpha(alphas_prime)
# print(optimizer.get_revenues_for_campaign(0))

optimizer.optimal_budget()

print("\n\nOptimizer2")

opt = Optimizer2(
    env.configuration.average_users_number,
    min_budget,
    max_budget,
    resolution,
    prices,
    quantities,
    alphas_prime,
    buy_probs)
opt.compute_rows()
opt.build_final_table()
opt.find_best_allocation()


if __name__ == '__main__':
    print("ok")
