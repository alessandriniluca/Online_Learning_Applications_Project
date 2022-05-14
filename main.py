import numpy as np
from product import Product
from environment import Environment


# TO DO: write correct different alpha functions
alphas = [1, 2, 4, 5, 7, 1, 2, 4, 9, 6, 7, 5, 3, 2, 1, 15]

def alphas_function(budget):
    increment = []
    for _ in range(3):
        for i in range(5):
            increment.append( (3.0 * (1.0 - np.exp(-.100*(budget[i])))).astype(int) )
    increment.append(0)
    return np.array(increment)

p1 = Product(name="P1", price=10, number=0)
p2 = Product(name="P2", price=15, number=1)
p3 = Product(name="P3", price=20, number=2)
p4 = Product(name="P4", price=5, number=3)
p5 = Product(name="P5", price=7, number=4)

p1.set_secondary(p2, p3)
p2.set_secondary(p3, p4)
p3.set_secondary(p1, p5)
p4.set_secondary(p1, p3)
p5.set_secondary(p1, p4)


products = [p1, p2, p3, p4, p5]

env = Environment(
    average_users_number=100, 
    std_users=100, 
    basic_alphas=alphas, 
    alphas_functions=alphas_function, 
    products=products, 
    lambda_prob=.8,
    graph_clicks=np.array([[.3, .2, .3, .5, .45], [.4, .3, .5, .6, .1], [.1, .1, .1, .2, .3], [.4, .2, .3, .3, .2], [.1, .2, .6, .8, .2]])
)

env.round(budget=[50, 10, 20, 10, 10])