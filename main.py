import numpy as np
from environment.product import Product
from environment.environment import Environment
from probability_calculator.probabilities import Probabilities
from statistics import NormalDist

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

calculator = Probabilities(env.graph_clicks, products, env.lambda_prob, env.reservation_price_means, env.reservation_price_std_dev)
test = calculator.get_buy_probs()
print(test)
print("p1 buy =")
p1_buy = 1-NormalDist(mu=9, sigma=2).cdf(10)
print(p1_buy)
p2_buy = 1-NormalDist(mu=14, sigma=3).cdf(15)
print("p2 buy = ")
print(p2_buy)
p3_buy = 1-NormalDist(mu=21, sigma=2).cdf(20)
print("p3 buy =")
print(p3_buy)
g_12 = env.graph_clicks[0][1]
print("graph_clicks p1->p2")
print(g_12)
g_23 = env.graph_clicks[1][2]
print("graph_clicks p2->p3")
print(g_23)
g_13 = env.graph_clicks[0][2]
print("graph_clicks p1->p3")
print(g_13)

print("If the following result is zero, then they are equal")
checksum = p1_buy*1*g_12*p2_buy*1*g_23*p3_buy + p1_buy*0.8*g_13*p3_buy-test[0][0][2]
print(checksum)
print("algorithm")
print(test[0][0][2])
print("obtained manually")
print(p1_buy*1*g_12*p2_buy*1*g_23*p3_buy + p1_buy*0.8*g_13*p3_buy)

print("second test, class 2")
# p4buy*0.8*p3_click*p3_buy +
# p4buy * 1 * p1_click * p1_buy * 0.8 *p3_click * p3_buy
# p4buy * 1 * p1_click * p1_buy * 1 * p2_ click * p2_buy * 1 * p3_click * p3_buy
p4buy = 1-NormalDist(mu=9, sigma=1).cdf(5)
p3buy = 1-NormalDist(mu=22, sigma=2).cdf(20)
p2buy = 1-NormalDist(mu=15, sigma=1).cdf(15)
p1buy = 1-NormalDist(mu=10, sigma=2).cdf(10)

click_4_3 = env.graph_clicks[3][2]
click_4_1 = env.graph_clicks[3][0]
click_1_3 = env.graph_clicks[0][2]
click_1_2 = env.graph_clicks[0][1]
click_2_3 = env.graph_clicks[1][2]

first_term = p4buy * 0.8 * click_4_3 * p3_buy
second_term = p4buy * 1 * click_4_1 * p1_buy * 0.8 * click_1_3 * p3_buy
third_term = p4buy * 1 * click_4_1 * p1_buy * 1 * click_1_2 * p2_buy * 1 * click_2_3 * p3_buy
prob_computed_algorithm = test[1][3][2]
prob_computed_manually = first_term + second_term + third_term
checksum = prob_computed_manually - prob_computed_algorithm

print("checksum")
print(checksum)
print("algorith")
print(prob_computed_algorithm)
print("manually")
print(prob_computed_manually)