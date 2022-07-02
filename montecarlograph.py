import numpy as np

from graph import Graph
from environment.product import Product

graph_clicks = np.array(
    [[.3, .2, .3, .5, .45], [.4, .3, .5, .6, .1], [.1, .1, .1, .2, .3], [.4, .2, .3, .3, .2], [.1, .2, .6, .8, .2]])
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

for i in range(5):
    g = Graph(
        click_probabilities=graph_clicks,
        products=products,
        user_buy_price_mean=np.array([9, 14, 21, 8, 4]),
        user_buy_price_std=np.array([2, 3, 2, 3, 3]),
        lambda_prob=0.8
    )
    g.simulate(starting_product_number=i, spin=1000000)
    res = g.get_results()
    tot = 0.
    for prod, prob in zip(products, res):
        tot += prod.price * prob
    print("Income:", tot)
    print(g)
    print()
