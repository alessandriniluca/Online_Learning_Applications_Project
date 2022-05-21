from pydoc import doc
import numpy as np
from environment.product import Product
from statistics import NormalDist

class Probabilities:
    def __init__(self, graph_clicks: np.ndarray, products: list, lambda_prob: float, reservation_price_means: np.ndarray, reservation_price_std_dev: np.ndarray):
        """init function of the class. N.b.: products needs to be ordered by number.

        Args:
            graph_clicks (np.ndarray): matrix of clicks: probability of starting from product (row) x, and clicking on product (column) y
            products (list): list containing all the products
            lambda_prob (float): probability of clicking on the second of the two secondary products
            reservation_price_means (np.ndarray): means of the reservation prices of the customers
            reservation_price_std_dev (np.ndarray): standard deviation of the reservation prices of the customers
        """
        self.graph_clicks = graph_clicks
        self.products = products
        self.lambda_prob = lambda_prob
        self.reservation_price_std_dev = reservation_price_std_dev
        self.reservation_price_means = reservation_price_means
    
    def prob_buy_starting_from(self, starting_prod:Product, prod:Product, user_class: int) -> float :
        buying_prob = 0
        queue = []
        buy_first_prob = 1 - (NormalDist(mu=self.reservation_price_means[user_class][starting_prod.number], sigma=self.reservation_price_std_dev[user_class][starting_prod.number]).cdf(starting_prod.price))
        queue.append([1, buy_first_prob, [], starting_prod])
        while queue:
            lamb, parent_buying_prob, viewed, current_prod = queue.pop()
            if current_prod in viewed:
                continue
            if current_prod == prod:
                buying_prob += parent_buying_prob
                continue
            viewed.append(current_prod)
            first_secondary, second_secondary = current_prod.get_secondaries()
            prob_buy_first = self.buy_prob_calculator(user_class, current_prod, first_secondary, parent_buying_prob, lamb)
            prob_buy_sec = self.buy_prob_calculator(user_class, current_prod, second_secondary, parent_buying_prob, lamb)
            queue.append([1, prob_buy_first, viewed.copy(), first_secondary])
            queue.append([self.lambda_prob, prob_buy_sec,viewed.copy(), second_secondary])
        return buying_prob
    
    def buy_prob_calculator(self, user_class:int, parent_prod:Product, child_prod:Product, parent_buying_prob:float, lamb:float)->float:
        buy_prob = 1 - (NormalDist(mu=self.reservation_price_means[user_class][child_prod.number], sigma=self.reservation_price_std_dev[user_class][child_prod.number]).cdf(child_prod.price))
        click_prob = self.graph_clicks[parent_prod.number][child_prod.number]
        prob = parent_buying_prob*lamb*click_prob*buy_prob
        return prob
    
    def get_buy_probs(self)->float:
        matrix = np.zeros((self.reservation_price_means.shape[0], len(self.products), len(self.products)))
        for user_class in range(self.reservation_price_means.shape[0]):
            for starting_prod in range(len(self.products)):
                for buying_prod in range(len(self.products)):
                    matrix[user_class][starting_prod][buying_prod] = self.prob_buy_starting_from(self.products[starting_prod], self.products[buying_prod], user_class)
        return matrix