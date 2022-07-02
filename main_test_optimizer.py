import numpy as np
from environment.product import Product
from environment.environment import Environment
from optimizer.optimizer import Optimizer
from optimizer.optimizer2 import Optimizer2
from probability_calculator.probabilities import Probabilities

# TO DO: write correct different alpha functions
alphas = np.array([[1, 1, 1, 1, 1, 100], [1, 1, 1, 1, 1, 80], [1, 1, 1, 1, 1, 90]])




def alphas_function_class_0(budget):
    increment = []
    increment.append( (30.0 * (1.0 - np.exp(-0.04*(budget[0])))).astype(int) )
    increment.append( (10.0 * (1.0 - np.exp(-0.035*(budget[1])))).astype(int) )
    increment.append( (15.0 * (1.0 - np.exp(-0.045*(budget[2])))).astype(int) )
    increment.append( (25.0 * (1.0 - np.exp(-0.04*(budget[3])))).astype(int) )
    increment.append( (60.0 * (1.0 - np.exp(-0.05*(budget[4])))).astype(int) )
    return np.array(increment)

def alphas_function_class_1(budget):
    increment = []
    increment.append( (25.0 * (1.0 - np.exp(-0.043*(budget[0])))).astype(int) )
    increment.append( (15.0 * (1.0 - np.exp(-0.039*(budget[1])))).astype(int) )
    increment.append( (10.0 * (1.0 - np.exp(-0.045*(budget[2])))).astype(int) )
    increment.append( (10.0 * (1.0 - np.exp(-0.03*(budget[3])))).astype(int) )
    increment.append( (70.0 * (1.0 - np.exp(-0.029*(budget[4])))).astype(int) )
        
    return np.array(increment)

def alphas_function_class_2(budget):
    increment = []
    increment.append( (30.0 * (1.0 - np.exp(-0.038*(budget[0])))).astype(int) )
    increment.append( (10.0 * (1.0 - np.exp(-0.045*(budget[1])))).astype(int) )
    increment.append( (50.0 * (1.0 - np.exp(-0.045*(budget[2])))).astype(int) )
    increment.append( (15.0 * (1.0 - np.exp(-0.044*(budget[3])))).astype(int) )
    increment.append( (45.0 * (1.0 - np.exp(-0.043*(budget[4])))).astype(int) )
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

np.set_printoptions(formatter={'float': lambda x: "{0:0.10f}".format(x)})

# TEST FOR OPTIMIZER
users_number=[5, 10, 4]
std_user_number = [2, 4, 2]
#alphas = np.array([[5, 6, 5, 3, 4, 18], [5, 6, 4, 4, 5, 16], [4, 4, 3, 5, 6, 20]])

env = Environment(
    average_users_number=users_number, 
    std_users=std_user_number,
    basic_alphas=alphas, 
    alphas_functions=alphas_functions, 
    products=products,
    lambda_prob=.8,
    graph_clicks=np.array([[.3, .2, .3, .5, .45], [.4, .3, .5, .6, .1], [.1, .8, .7, .2, .3], [.4, .7, .8, .3, .2], [.1, .8, .6, .8, .2]])
  )


calculator = Probabilities(env.graph_clicks, products, env.lambda_prob, env.reservation_price_means, env.reservation_price_std_dev)
buy_probs = calculator.get_buy_probs()
quantities = env.quantity_means
prices = [80, 34, 99, 51, 23]


total_budget = 180
resolution = 5
min_budget = [0,0,0,0,0]
max_budget = [180, 180, 180, 180, 180]

alphas_prime = np.zeros((int(total_budget/resolution)+1, len(products), 3))

for single_budget in range(0, total_budget+resolution, resolution):
    for product_index in range(len(products)):
        for i, users in enumerate(users_number):
            # set budget to corresponding product (using array with zeros for alpha function compatibility)
            budgets = np.zeros(5)
            budgets[product_index] = single_budget
            # compute deltas of weights
            delta_alpha_weights = alphas_functions[i](budgets)
            # concatenate a zero in order to consider also lost visitors
            delta_alpha_weights = np.concatenate((delta_alpha_weights, np.array([0])))
            # compute alpha primes (expected value of dirichlet variables)
            alphas_prime[int(single_budget/resolution)][product_index][i] = (alphas[i] + delta_alpha_weights)[product_index]/sum((alphas[i] + delta_alpha_weights))
print("EEEEEEE", alphas_prime, "FFFFFFFF")

optimizer = Optimizer(users_number, min_budget, max_budget, buy_probs, prices, total_budget, resolution, quantities)
optimizer.set_alpha(alphas_prime)
# print(optimizer.get_revenues_for_campaign(0))

optimizer.optimal_budget()


print("\n\nOptimizer2")

opt = Optimizer2(users_number, min_budget, max_budget, resolution, prices, quantities, alphas_prime, buy_probs)
opt.compute_rows()
opt.build_final_table()
opt.find_best_allocation()