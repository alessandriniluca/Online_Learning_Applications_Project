from pydoc import doc
from unicodedata import decimal
import numpy as np
# from environment.product import Product
from statistics import NormalDist
from decimal import Decimal

class Optimizer:

    def __init__(self, users_number: list, min_budget: list, max_budget: list, alphas, alphas_functions, buy_probs, prices, total_budget, resolution):
        # TODO better to have an array of functions with 3 function (one per category as alpha functions)
        # ALPHA FUNCTION HANDLER?
        self.users_number = users_number
        self.alphas = alphas
        self.alphas_functions = alphas_functions
        self.buy_probs = buy_probs
        self.prices = prices
        self.total_budget = total_budget
        self.resolution = resolution
        self.min_budget = min_budget
        self.max_budget = max_budget
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
            if(budget>=self.min_budget[product_index] and budget<=self.max_budget[product_index]):
                revenues.append(self.get_revenue(product_index,budget))
            else:
                revenues.append(-np.inf)
        revn = [[-np.inf, 90, 100, 105, 110, -np.inf, -np.inf, -np.inf],[0, 82, 90, 92, -np.inf,-np.inf,-np.inf,-np.inf],[0,80,83,85,86,-np.inf,-np.inf,-np.inf],[-np.inf,90,110,115,118,120,-np.inf,-np.inf],[-np.inf,111,130,138,142,148,155,-np.inf]]
        return revn[product_index]
        return revenues

    def optimal_budget_matrix(self):
        table = np.zeros((5+1, int(self.total_budget/self.resolution)+1))
        history = np.zeros((5+1, int(self.total_budget/self.resolution)+1))
        for i in range(1,6):
            revenues_current_campain = self.get_revenues_for_campaign(i-1)
            for j in range(int(self.total_budget/self.resolution)+1):
                if(revenues_current_campain[j] != -np.inf):
                    table[i][j] = max((revenues_current_campain[k]+table[i-1][j-k]) for k in range(j+1)) - self.resolution*j
                    history[i][j] = int(np.argmax(np.asarray([(revenues_current_campain[k]+table[i-1][j-k]) for k in range(j+1)])))
                else:
                    table[i][j] = -np.inf
        np.set_printoptions(precision=2)
        print(np.asarray([i for i in range(0,self.total_budget+self.resolution, self.resolution)])) # TODO: anche la funzione sopra non tiene conto del total_budget perchè il range si ferma lo step prima del total_budget, quindi bisogna aggiungere uno step
        print(table)
        # print(history)
        return table, history

    # k è uguale al'indice del budget di quanto abbiamo speso nella campagna corrente i nostri soldi
    def resolver(self, j, k):
        arrayBudgets = np.asarray([i for i in range(0,self.total_budget+self.resolution, self.resolution)])
        return arrayBudgets[k], arrayBudgets[j-k]
        
    def optimal_budget(self):
        table, history = self.optimal_budget_matrix()
        optimal = np.zeros((5))
        j = np.argmax(table[5])
        k = int(history[5][j])
        optimal[4], tempMax = self.resolver(j, k)
        for i in range(4,0,-1):
            j = int(tempMax/self.resolution)
            k = int(history[i][j])
            optimal[i-1], tempMax = self.resolver(j, k)
        print("Optimal: ")
        print(optimal)

