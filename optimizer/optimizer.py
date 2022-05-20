from pydoc import doc
import numpy as np
# from environment.product import Product
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
        assert total_budget%resolution==0, "il budget e la risoluzione non sono divisibili" # TODO

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

    # per ogni cella della campagna devo prendere il 
    # max{(c_i+c_i-1), ...} di tutti i budget dall'inizio a quella cella
    # 
    def func(self):
        table = np.zeros((5+1, int(self.total_budget/self.resolution)))
        for i in range(1,6):
            for j in range(int(self.total_budget/self.resolution)):
                revenues_current_campain = self.get_revenues_for_campaign(i-1)
                print(revenues_current_campain)
                print()
                table[i][j] = max((revenues_current_campain[k]+table[i-1][j-k]) for k in range(j+1))
                # print(table)