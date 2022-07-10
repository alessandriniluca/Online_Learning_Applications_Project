import numpy as np
from to_be_evaluated.MAB_step_3.GPTS_Alphas import GPTS_Alphas
from environment.product import Product
from environment.environment import Environment
from optimizer.optimizer2 import Optimizer2
from probability_calculator.probabilities import Probabilities
from matplotlib import pyplot as plt

from step_4.quantities_estimator import QuantitiesEstimator

np.set_printoptions(formatter={'float': lambda x: "{0:0.10f}".format(x)})


# TO DO: write correct different alpha functions


def alphas_function_class_0(budget):
    increment = []
    increment.append((30.0 * (1.0 - np.exp(-0.04 * (budget[0])))).astype(int))
    increment.append((10.0 * (1.0 - np.exp(-0.035 * (budget[1])))).astype(int))
    increment.append((15.0 * (1.0 - np.exp(-0.045 * (budget[2])))).astype(int))
    increment.append((25.0 * (1.0 - np.exp(-0.04 * (budget[3])))).astype(int))
    increment.append((60.0 * (1.0 - np.exp(-0.05 * (budget[4])))).astype(int))
    return np.array(increment)


def alphas_function_class_1(budget):
    increment = []
    increment.append((25.0 * (1.0 - np.exp(-0.043 * (budget[0])))).astype(int))
    increment.append((15.0 * (1.0 - np.exp(-0.039 * (budget[1])))).astype(int))
    increment.append((10.0 * (1.0 - np.exp(-0.045 * (budget[2])))).astype(int))
    increment.append((10.0 * (1.0 - np.exp(-0.03 * (budget[3])))).astype(int))
    increment.append((70.0 * (1.0 - np.exp(-0.029 * (budget[4])))).astype(int))

    return np.array(increment)


def alphas_function_class_2(budget):
    increment = []
    increment.append((30.0 * (1.0 - np.exp(-0.038 * (budget[0])))).astype(int))
    increment.append((10.0 * (1.0 - np.exp(-0.045 * (budget[1])))).astype(int))
    increment.append((50.0 * (1.0 - np.exp(-0.045 * (budget[2])))).astype(int))
    increment.append((15.0 * (1.0 - np.exp(-0.044 * (budget[3])))).astype(int))
    increment.append((45.0 * (1.0 - np.exp(-0.043 * (budget[4])))).astype(int))
    return np.array(increment)


alphas_functions = [alphas_function_class_0, alphas_function_class_1, alphas_function_class_2]

p1 = Product(name="P0", price=80, number=0)
p2 = Product(name="P1", price=34, number=1)
p3 = Product(name="P2", price=99, number=2)
p4 = Product(name="P3", price=51, number=3)
p5 = Product(name="P4", price=23, number=4)

p1.set_secondary(p2, p3)
p2.set_secondary(p3, p4)
p3.set_secondary(p1, p5)
p4.set_secondary(p1, p3)
p5.set_secondary(p1, p4)

products = [p1, p2, p3, p4, p5]
alphas = np.array([[1, 1, 1, 1, 1, 50], [1, 1, 1, 1, 1, 30], [1, 1, 1, 1, 1, 30]])
users_number = [5, 10, 4]
std_user_number = [2, 4, 2]

prices = [p1.price, p2.price, p3.price, p4.price, p5.price]

total_budget = 180
resolution = 5
min_budget = [0, 0, 0, 0, 0]
max_budget = [180, 180, 180, 180, 180]

spaces = []
for bmin, bmax in zip(min_budget, max_budget):
    spaces.append(np.linspace(int(bmin), int(bmax), num=int(total_budget / resolution + 1)))

time = 50
n_experiments = 4000
reward_per_experiment = []

for e in range(0, n_experiments):
    print("experiment:", e)
    env = Environment(
        average_users_number=users_number,
        std_users=std_user_number,
        basic_alphas=alphas,
        alphas_functions=alphas_functions,
        products=products,
        lambda_prob=.8,
        graph_clicks=np.array([[.3, .2, .3, .5, .45], [.4, .3, .5, .6, .1], [.1, .8, .7, .2, .3], [.4, .7, .8, .3, .2],
                               [.1, .8, .6, .8, .2]])
    )

    calculator = Probabilities(env.graph_clicks, products, env.lambda_prob, env.reservation_price_means,
                               env.reservation_price_std_dev)
    buy_probs = calculator.get_buy_probs()

    ts_learners = GPTS_Alphas([int(total_budget / resolution + 1)] * 5, spaces)
    quantity_estimator = QuantitiesEstimator(products)
    reward_per_round = []

    for t in range(time):
        print("step:", t)
        alphas_prime = ts_learners.pull_arms()
        quantities = quantity_estimator.get_quantities()
        # print("SHAPE-----------", alphas_prime, alphas_prime.shape)

        opt = Optimizer2(users_number, min_budget, max_budget, resolution, prices, quantities, alphas_prime, buy_probs)
        opt.compute_rows()
        opt.build_final_table()
        budget = opt.find_best_allocation()

        # optimizer = Optimizer([sum(users_number)], min_budget, max_budget, buy_probs, prices, total_budget, resolution, quantities)
        # optimizer.set_alpha(alphas_prime)
        # budget = optimizer.optimal_budget()
        print("Estimated users:", sum(users_number))

        rewards, total_users = env.round(budget)

        total_earning = 0

        for user in rewards:
            for bought in user.bought_product:
                total_earning += bought[0].price * bought[1]

        reward_per_round.append(total_earning - int(sum(budget)))
        print("Env money:", total_earning - int(sum(budget)))

        idx_played_arms = []
        for b in budget:
            idx_played_arms.append(np.where(b == np.array(spaces[0]))[0][0])

        starting_from = []
        for reward in rewards:
            starting_from.append(reward.starting_product)

        for i in range(total_users - len(rewards)):
            starting_from.append(6)

        ts_learners.update(idx_played_arms, starting_from)
        quantity_estimator.update_quantities(rewards)

    reward_per_experiment.append(reward_per_round)
    print("-----------TOTALE", np.mean(reward_per_experiment, axis=0))

print(reward_per_experiment)

rewards = np.array(reward_per_experiment)

print(np.mean(rewards, axis=0))
x = (np.linspace(start=0, stop=9, num=10))
plt.figure(0)
# plt.title(f'Iteration {self.t} {self.name}')
plt.plot(x, np.mean(rewards, axis=0), 'ro', label=r'Observed Clicks')
plt.show()
