import numpy as np
from environment.product import Product
from environment.environment import Environment
from optimizer.optimizer import Optimizer

# TO DO: write correct different alpha functions
alphas = np.array([[1, 2, 4, 5, 7, 15], [1, 2, 4, 9, 6, 16], [7, 5, 3, 2, 1, 15]])

def alphas_function(budget):
    increment = []
    for _ in range(3):
        row = []
        for i in range(5):
            row.append( (3.0 * (1.0 - np.exp(-.100*(budget[i])))).astype(int) )
        row.append(0)
        increment.append(row)
    return np.array(increment)

p1 = Product(name="P0", price=10, number=0)
p2 = Product(name="P1", price=15, number=1)
p3 = Product(name="P2", price=20, number=2)
p4 = Product(name="P3", price=5, number=3)
p5 = Product(name="P4", price=7, number=4)

p1.set_secondary(p2, p3)
p2.set_secondary(p3, p4)
p3.set_secondary(p1, p5)
p4.set_secondary(p1, p3)
p5.set_secondary(p1, p4)


products = [p1, p2, p3, p4, p5]

env = Environment(
    average_users_number=[100, 130, 110], 
    std_users=[10, 15, 8],
    basic_alphas=alphas, 
    alphas_functions=alphas_function, 
    products=products, 
    lambda_prob=.8,
    graph_clicks=np.array([[.3, .2, .3, .5, .45], [.4, .3, .5, .6, .1], [.1, .1, .1, .2, .3], [.4, .2, .3, .3, .2], [.1, .2, .6, .8, .2]])
)

env.round(budget=[50, 10, 20, 10, 10])

np.set_printoptions(formatter={'float': lambda x: "{0:0.10f}".format(x)})

# TEST FOR OPTIMIZER
users_number=[100, 130, 110]
alphas = np.array([[1, 2, 4, 5, 7, 15], [1, 2, 4, 9, 6, 16], [7, 5, 3, 2, 1, 15]])

def alphas_function_new(budget):
    increment = []
    for i in range(5):
        increment.append( (3.0 * (1.0 - np.exp(-.1*(budget[i])))).astype(int) )
    return np.array(increment)

alphas_functions = [alphas_function_new, alphas_function_new, alphas_function_new]

buy_probs = test
prices = [10, 15, 20, 5, 7]

total_budget = 100
resolution = 10

optimizer = Optimizer(users_number, alphas, alphas_functions, buy_probs, prices, total_budget, resolution)
print(optimizer.get_revenues_for_campaign(0))
