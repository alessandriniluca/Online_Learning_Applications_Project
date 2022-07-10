import numpy as np

from environment.product import Product


class User:
    def __init__(self, reservation_price_means, reservation_price_std_dev, quantity_means, quantity_std_dev,
                 user_class=None, features=None, starting_product=1, graph_clicks=None):
        self.reservation_price = np.random.normal(reservation_price_means, reservation_price_std_dev)
        self.quantity_means = quantity_means
        self.quantity_std_dev = quantity_std_dev
        self.seen_product = []
        self.bought_product = []
        self.user_class = user_class
        self.features = features
        self.starting_product = starting_product
        self.graph_clicks = graph_clicks

    def has_bought(self, product):
        return self.reservation_price[product.number] > product.price

    def quantity_bought(self, product):
        return int(
            max(1., (np.random.normal(self.quantity_means[product.number], self.quantity_std_dev[product.number]))))

    def product_click(self, main_prod: Product, secondary_prod: Product):
        """
        Return true if user that has landed in main click on secondary_prod

        """
        return np.random.uniform(0.0, 1.0) < self.graph_clicks[main_prod.number][secondary_prod.number]

    def add_seen_product(self, product):
        self.seen_product.append(product)
        self.graph_clicks[:, product.number] = 0

    def add_bought_product(self, product, quantity):
        self.bought_product.append([product, quantity])
