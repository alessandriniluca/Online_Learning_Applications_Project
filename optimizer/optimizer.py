from pydoc import doc
import numpy as np
from environment.product import Product
from statistics import NormalDist

class Optimizer:

    def __init__(self, users_number: list, alphas, alphas_functions, buy_probs, prices, total_budget, resolution):
        # TODO better to have an array of functions with 3 function (one per category as alpha functions)
        # ALPHA FUNCTION HANDLER?
        self.users_number = users_number
        self.alphas = alphas
        self.alphas_functions = alphas_functions
        self.buy_probs = buy_probs
        self.prices = prices
        self.total_budget = total_budget
        self.resolution = resolution

    def get_revenue(self, product_index, single_budget):
        total_revenue = 0
        for i, users in enumerate(self.users_number):
            budgets = np.zeros(5)
            budgets[product_index] = single_budget
            alphas_prime = self.alphas[i][:5] + self.alphas_functions[i](budgets)
            weighted_price = 0
            # Lo so è un po' sus
            for k in range(5):
                weighted_price += self.buy_probs[i][product_index][k] * self.prices[k]

            revenue = users * alphas_prime[product_index] * weighted_price
            total_revenue += revenue
        return total_revenue

    def get_revenues_for_campaign(self, product_index):
        revenues = []
        for budget in range(0,self.total_budget, self.resolution):
            revenues.append(self.get_revenue(product_index,budget))
        return revenues





