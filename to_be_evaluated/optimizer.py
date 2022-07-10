from pydoc import doc
from unicodedata import decimal
import numpy as np
# from environment.product import Product
from statistics import NormalDist
from decimal import Decimal


class Optimizer:

    def __init__(self, users_number: list, min_budget: list, max_budget: list, buy_probs, prices, total_budget,
                 resolution, quantities):
        # TODO better to have an array of functions with 3 function (one per category as alpha functions)
        # ALPHA FUNCTION HANDLER?
        """Optimizer class

        Args:
            users_number (list): number of users per category
            min_budget (list): minimum budget constraints per campaign
            max_budget (list): maximum budget constraints per campaign
            alphas (np.array): alpha array (note: contains weights of dirichlet)
            alphas_functions (list): alpha functions pointers (one per category)
            buy_probs (np.array): buy probability matrix
            prices (list): item prices
            total_budget (int): total budget for all campaings
            resolution (int): minimum spendable amount
        """
        self.users_number = users_number
        self.buy_probs = buy_probs
        self.prices = prices
        self.total_budget = total_budget
        self.resolution = resolution
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.quantities = quantities
        assert total_budget % resolution == 0, "il budget e la risoluzione non sono divisibili"  # TODO

    def set_alpha(self, alpha):
        self.alpha_prime = alpha

    def get_revenue(self, product_index, single_budget):
        """Expected revenue from a budget allocation

        Args:
            product_index (_type_): product index of associated campaign
            single_budget (_type_): budget to assign to that campaign
 
        Returns:
            _type_: expected revenue
        """
        total_revenue = 0
        # for each user category
        for i, users in enumerate(self.users_number):
            # set budget to corresponding product (using array with zeros for alpha function compatibility)
            # budgets = np.zeros(5)
            # budgets[product_index] = single_budget
            # # compute deltas of weights
            # delta_alpha_weights = self.alphas_functions[i](budgets)
            # # concatenate a zero in order to consider also lost visitors
            # delta_alpha_weights = np.concatenate((delta_alpha_weights, np.array([0])))
            # # compute alpha primes (expected value of dirichlet variables)
            # alphas_prime = self.alphas[i] + delta_alpha_weights
            # alphas_prime = alphas_prime/sum(alphas_prime)

            # In order to compute expected revenue we have to sum
            # Users * alpha_prime * (sum of buy prob of product A given starting product B * price of A)

            # we first compute all the sums of price * buy prob
            weighted_price = 0
            for k in range(len(self.prices)):
                weighted_price += self.buy_probs[i][product_index][k] * self.prices[k] * self.quantities[i][k]

            # Then we multiply with users and alpha_prime
            # print("user:", i , "QQQQQQQ", users, int(users * self.alpha_prime[int(single_budget/self.resolution)][product_index][i]), users * (self.alpha_prime[int(single_budget/self.resolution)][product_index][i]))
            revenue = int(
                users * self.alpha_prime[int(single_budget / self.resolution)][product_index][i]) * weighted_price

            # Sum with all user categories
            total_revenue += revenue
        return total_revenue

    def get_revenues_for_campaign(self, product_index):
        """compute all revenues for a single campaign

        Args:
            product_index (_type_): product index of associated campaign

        Returns:
            list: all revenues
        """
        revenues = []
        # for each possible budget
        for budget in range(0, self.total_budget + self.resolution, self.resolution):
            # check constraints
            if (budget >= self.min_budget[product_index] and budget <= self.max_budget[product_index]):
                # if check successful compute single revenue
                revenues.append(self.get_revenue(product_index, budget))
            else:
                # else append -inf
                revenues.append(-np.inf)

        # revn = [[-np.inf, 50, 51, 58, 61, -np.inf],[-np.inf, 29, 31, 35, 44, 48],[-np.inf, 45, 55, 58, 62, 68],[-np.inf,38, 44, 49, 58, 61],[-np.inf,42, 48, 49, 53, 59]]
        # return revn[product_index]
        return revenues

    def optimal_budget_matrix(self):
        """generate optimal budget matrix

        Returns:
            _type_: optimal budget matrix
        """
        # output matrix
        table = np.zeros((5 + 1, int(self.total_budget / self.resolution) + 1))
        # for each cell store position of best budget allocation in subrow
        history = np.zeros((5 + 1, int(self.total_budget / self.resolution) + 1))

        for i in range(1, 6):
            # Compute all revenues for a single campaign
            revenues_current_campain = self.get_revenues_for_campaign(i - 1)
            # For each budget allocation
            for j in range(int(self.total_budget / self.resolution) + 1):
                if (revenues_current_campain[j] != -np.inf):
                    # choose max between all possible budget allocation with predecessors
                    table[i][j] = max((revenues_current_campain[k] + table[i - 1][j - k]) for k in
                                      range(j + 1))  # - self.resolution*j
                    # store decision taken (index of revenues in order to reconstruct history at the end)
                    history[i][j] = int(np.argmax(
                        np.asarray([(revenues_current_campain[k] + table[i - 1][j - k]) for k in range(j + 1)])))
                else:
                    table[i][j] = -np.inf

        np.set_printoptions(precision=2)
        # ee print(np.asarray([i for i in range(0,self.total_budget+self.resolution, self.resolution)]))
        # ee print(table)
        # print(history)
        return table, history

    # k è uguale al'indice del budget di quanto abbiamo speso nella campagna corrente
    def resolver(self, j, k):
        """resolve j k history to get information about how the budget has been allocated

        Args:
            j (_type_): index of the budget
            k (_type_): index of max cell

        Returns:
            _type_: (current campaign allocation, all predecessor campaings budget)
        """
        arrayBudgets = np.asarray([i for i in range(0, self.total_budget + self.resolution, self.resolution)])
        return arrayBudgets[k], arrayBudgets[j - k]

    def optimal_budget(self):
        """get optimal budget allocations
        """
        table, history = self.optimal_budget_matrix()

        optimal = np.zeros((5))
        # index of optimal budget picked from last line
        j = np.argmax(table[5] - np.array([i for i in range(0, self.total_budget + self.resolution, self.resolution)]))
        # index of max cell from last line
        k = int(history[5][j])
        # first unroll operation
        optimal[4], tempMax = self.resolver(j, k)
        # for each campaign unroll and save best budget allocations
        for i in range(4, 0, -1):
            j = int(tempMax / self.resolution)
            k = int(history[i][j])
            optimal[i - 1], tempMax = self.resolver(j, k)
        print("Optimal: ", optimal)
        print("Money:",
              np.max(table[5] - np.array([i for i in range(0, self.total_budget + self.resolution, self.resolution)])))
        return optimal
