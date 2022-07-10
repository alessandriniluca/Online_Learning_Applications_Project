import queue
from statistics import NormalDist
import numpy as np


class Graph:
    def __init__(self, click_probabilities=None, products=None, reservation_price_means=None,
                 reservation_price_std_dev=None, lambda_prob=1):
        """ Initialize the graph

        Args:
            click_probabilities (numpy matrix) : Defaults to None.
            products (numpy array): The products. Defaults to None.
            reservation_price_means (numpy array): Mean of reservation price for a single class of users. Defaults to None.
            reservation_price_std_dev (numpy array): Std dev of reservation price for a single class of users. Defaults to None.
            lambda_prob (int): . Defaults to 1.
        """
        self.click_probabilities = click_probabilities
        self.products = products
        self.reservation_price_means = reservation_price_means
        self.reservation_price_std_dev = reservation_price_std_dev
        self.actual_node = None
        self.activations = [0, 0, 0, 0, 0]
        self.rounds = 0
        self.lambda_prob = lambda_prob
        self.queue = []
        self.starting = None

    def set_starting(self, product_number=1):
        """Set the starting node before the simulation round starts

        Args:
            product_number (int, optional): Number of the starting product, the seed. Defaults to 1.
        """
        self.starting = self.products[product_number]
        self.actual_node = self.products[product_number]

    def simulate_round(self):
        """Simulate a single iteration of Monte Carlo simulation
        """
        self.rounds += 1
        self.queue = []
        self.queue.append(self.actual_node)
        copy_click_probabilities = self.click_probabilities.copy()

        while len(self.queue) > 0:
            self.actual_node = self.queue[0]
            mean = self.reservation_price_means[self.actual_node.number]
            std = self.reservation_price_std_dev[self.actual_node.number]
            if (np.random.normal(mean, std) > self.actual_node.price):
                copy_click_probabilities[:, self.actual_node.number] = 0
                self.activations[self.actual_node.number] += 1
                prob_click_a = copy_click_probabilities[self.actual_node.number][self.actual_node.secondary_a.number]
                prob_click_b = self.lambda_prob * copy_click_probabilities[self.actual_node.number][
                    self.actual_node.secondary_b.number]
                v = np.random.uniform(.0, 1.)
                if (v < prob_click_a):
                    self.queue.append(self.actual_node.secondary_a)
                v = np.random.uniform(.0, 1.)
                if (v < prob_click_b):
                    self.queue.append(self.actual_node.secondary_b)
            self.queue = self.queue[1:]

    def simulate(self, starting_product_number=1, spin=10000):
        """Simulate K iterations of Monte Carlo algorithm 

        Args:
            starting_product_number (int, optional): Seed number. Defaults to 1.
            spin (int, optional): Number of iterations. Defaults to 10000.
        """
        self.set_starting(product_number=starting_product_number)
        for i in range(spin):
            self.simulate_round()
            self.set_starting(product_number=starting_product_number)

    def get_results(self):
        return np.array(self.activations) / self.rounds

    def __str__(self):
        out = ""
        for i in range(len(self.products)):
            out += "Probability buy product P" + str(i) + " starting from P" + str(self.starting.number) + ": " + str(
                self.activations[i] / self.rounds) + "\n"
        return out

    def simulate_round_exact_prob_to_buy(self):
        """ Method that can be used as alternative to simulate_round, the difference is that the buy probability is computed exactly
        """
        self.rounds += 1
        node_list = []
        graph = self.click_probabilities.copy()
        node_list.append(self.actual_node)

        while len(node_list) > 0:
            self.actual_node = node_list[0]
            graph[:, self.actual_node.number] = 0
            buy_first_prob = 1 - (NormalDist(mu=self.reservation_price_means[self.actual_node.number],
                                             sigma=self.reservation_price_std_dev[self.actual_node.number]).cdf(
                self.actual_node.price))
            if np.random.uniform(0, 1) < buy_first_prob:
                self.activations[self.actual_node.number] += 1

                secondary_a = self.actual_node.secondary_a
                secondary_b = self.actual_node.secondary_b
                if np.random.uniform(0, 1) < graph[self.actual_node.number][secondary_a.number]:
                    node_list.append(secondary_a)

                if np.random.uniform(0, 1) < graph[self.actual_node.number][secondary_b.number] * self.lambda_prob:
                    node_list.append(secondary_b)

            node_list = node_list[1:]
