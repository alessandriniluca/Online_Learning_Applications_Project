import numpy as np
from environment.product import Product
from environment.environment import Environment
from optimizer.estimator import Estimator

# TO DO: write correct different alpha functions
alphas = np.array([[1, 2, 4, 5, 7, 15], [1, 2, 4, 9, 6, 16], [7, 5, 3, 2, 1, 15]])


def alphas_function_class_1(budget):
    increment = []
    for i in range(5):
        increment.append( (7.0 * (1.0 - np.exp(-0.02*(budget[i])))).astype(int) )
    return np.array(increment)


alphas_functions = [alphas_function_class_1, alphas_function_class_1, alphas_function_class_1]

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
    alphas_functions=alphas_functions,
    products=products, 
    lambda_prob=.8,
    graph_clicks=np.array([[.3, .2, .3, .5, .45], [.4, .3, .5, .6, .1], [.1, .1, .1, .2, .3], [.4, .2, .3, .3, .2], [.1, .2, .6, .8, .2]])
)

env.round(budget=[50, 10, 20, 10, 10])

np.set_printoptions(formatter={'float': lambda x: "{0:0.10f}".format(x)})

calculator = Estimator(env.graph_clicks, products, env.lambda_prob, env.reservation_price_means, env.reservation_price_std_dev)
probabilities_matrix = calculator.get_buy_probs()
print(probabilities_matrix)

print("\n\n")
probabilities_matrix_montacarlo = calculator.montecarlo_get_buy_probs()
print(probabilities_matrix_montacarlo)




# compute the error

tot_1 = 0
tot_2 = 0
error = np.zeros((3, 5, 5))
for i in range(3):
    for j in range(5):
        tot_1 += probabilities_matrix[i][j]
        tot_2 += probabilities_matrix_montacarlo[i][j]
        error[i][j] = probabilities_matrix[i][j] - probabilities_matrix_montacarlo[i][j]

print("\n\n\n", error)
print("\n\n tot:", sum(error.flatten()))