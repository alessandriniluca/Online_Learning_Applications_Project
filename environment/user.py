import numpy as np


class User:
    def __init__(self, reservation_price_means, reservation_price_std_dev, quantity_means, quantity_std_dev,
                 user_class=None, features=[], starting_product=1):
        self.reservation_price = np.random.normal(reservation_price_means, reservation_price_std_dev)
        self.quantity_means = quantity_means
        self.quantity_std_dev = quantity_std_dev
        self.seen_product = []
        self.bought_product = []
        self.user_class = user_class
        self.features = features
        self.starting_product = starting_product

    def has_bought(self, product, stampa=False):
        return self.reservation_price[product.number] > product.price

    def quantity_bought(self, product):
        return int(
            max(1., (np.random.normal(self.quantity_means[product.number], self.quantity_std_dev[product.number]))))

    def add_seen_product(self, product):
        self.seen_product.append(product)

    def add_bought_product(self, product, quantity):
        self.bought_product.append([product, quantity])
