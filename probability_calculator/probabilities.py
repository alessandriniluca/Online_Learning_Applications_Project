from pydoc import doc
import numpy as np
from environment.product import Product
from statistics import NormalDist

from graph import Graph


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
        """Computing the probability of buying product prod given that our navigation started from product starting_prod, according to the specified user class

        Args:
            starting_prod (Product): product from which our navigation starts
            prod (Product): product of which we want to compute the probability of buying, starting the navigation from starting_prod
            user_class (int): class of the user considered. In the standard version of the problem they are three

        Returns:
            float: probability of buying prod given that the navigation of the user of class user_class started with starting_prod
        """
        buying_prob = 0
        not_buying_prob = 1
        queue = []
        buy_first_prob = 1 - (NormalDist(mu=self.reservation_price_means[user_class][starting_prod.number], sigma=self.reservation_price_std_dev[user_class][starting_prod.number]).cdf(starting_prod.price))
        queue.append([buy_first_prob, [], starting_prod])
        while queue:
            parent_buying_prob, viewed, current_prod = queue.pop()
            if current_prod in viewed:
                continue
            if current_prod == prod:
                buying_prob += not_buying_prob * parent_buying_prob
                not_buying_prob *= (1 - parent_buying_prob)
                continue
            viewed.append(current_prod)
            first_secondary, second_secondary = current_prod.get_secondaries()
            prob_buy_first = self.buy_prob_calculator(user_class, current_prod, first_secondary, parent_buying_prob, 1)
            prob_buy_sec = self.buy_prob_calculator(user_class, current_prod, second_secondary, parent_buying_prob, self.lambda_prob)
            queue.append([prob_buy_first, viewed.copy(), first_secondary])
            queue.append([prob_buy_sec, viewed.copy(), second_secondary])
        return buying_prob
    
    def buy_prob_calculator(self, user_class:int, parent_prod:Product, child_prod:Product, parent_buying_prob:float, lamb:float)->float:
        """Computing the probability of buying one of the two child products (child_prod) given the probability of buyng its parent (parent_buying_prob), according to the class of the user

        Args:
            user_class (int): class of the user
            parent_prod (Product): main product in the page
            child_prod (Product): one of the two child products of the page
            parent_buying_prob (float): probability of buying the parent
            lamb (float): probability to see the child_prod (in the standard version of the assignment is 1 for the first secondary products, and 0.8 for the second secondary product)

        Returns:
            float: probability of buying the secondary product child_progt given the probability of buying parent_prod, specified as parent_buying_prod (according to the specific user class)
        """
        buy_prob = 1 - (NormalDist(mu=self.reservation_price_means[user_class][child_prod.number], sigma=self.reservation_price_std_dev[user_class][child_prod.number]).cdf(child_prod.price))
        click_prob = self.graph_clicks[parent_prod.number][child_prod.number]
        prob = parent_buying_prob*lamb*click_prob*buy_prob
        return prob
    
    def get_buy_probs(self)->float:
        """This function comptues the probability of buying every product starting from every product, according to the class of the user. The starting product is the product with which the navigation starts

        Returns:
            float: is a matrix of floats (probabilities), with indexes: [class of the user][starting product][buying product]
        """
        matrix = np.zeros((self.reservation_price_means.shape[0], len(self.products), len(self.products)))
        for user_class in range(self.reservation_price_means.shape[0]):
            for starting_prod in range(len(self.products)):
                for buying_prod in range(len(self.products)):
                    matrix[user_class][starting_prod][buying_prod] = self.prob_buy_starting_from(self.products[starting_prod], self.products[buying_prod], user_class)
        return matrix


    def montecarlo_get_buy_probs(self):
        """This method compute the probabilities of buying a product starting from another, using Monte Carlo simulation

        Returns:
            float: is a matrix of floats (probabilities), with indexes: [class of the user][starting product][buying product]
        """
        result = []
        for user_class in range(self.reservation_price_means.shape[0]):
            class_result = []
            for starting_prod in range(len(self.products)):
                g = Graph(
                    click_probabilities=self.graph_clicks,
                    products=self.products, 
                    reservation_price_means=self.reservation_price_means[user_class],
                    reservation_price_std_dev=self.reservation_price_std_dev[user_class],
                    lambda_prob=self.lambda_prob
                )
                g.simulate(starting_product_number=starting_prod, spin=10000)
                class_result.append(g.get_results().tolist())
            result.append(class_result)
        return np.array(result)