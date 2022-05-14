import numpy as np

class Product:
    def __init__(self, name, price, number):
        self.name = name
        self.price = price
        self.number = number

    def set_secondary(self, prod_a, prod_b):
        self.secondary_a = prod_a
        self.secondary_b = prod_b