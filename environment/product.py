import numpy as np


class Product:
    """
    Base class for product
    """
    def __init__(self, name, price, number):
        self.name = name
        self.price = price
        self.number = number
        self.secondary_a = None
        self.secondary_b = None

    def set_secondary(self, prod_a, prod_b):
        self.secondary_a = prod_a
        self.secondary_b = prod_b

    def __str__(self):
        return "P" + str(self.number)

    def get_secondaries(self):
        return self.secondary_a, self.secondary_b

    def __eq__(self, other):
        return self.name == other.name
