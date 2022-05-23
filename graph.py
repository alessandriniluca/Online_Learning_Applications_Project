import numpy as np

class Graph:
    def __init__(self, click_probabilities=None, products=None, reservation_price_means=None, reservation_price_std_dev=None, lambda_prob=1):
        self.click_probabilities = click_probabilities
        self.products = products
        self.reservation_price_means = reservation_price_means
        self.reservation_price_std_dev = reservation_price_std_dev
        self.actual_node = None
        self.activations = [0, 0, 0, 0 ,0]
        self.rounds = 0
        self.lambda_prob = lambda_prob
        self.queue = []
        self.starting = None

    def set_starting(self, product_number=1):
        self.starting = self.products[product_number]
        self.actual_node = self.products[product_number]

    def simulate_round(self):
        self.rounds += 1
        product_number = self.actual_node.number
        mean = self.reservation_price_means[product_number]
        std = self.reservation_price_std_dev[product_number]
        self.queue = []
        self.queue.append(self.actual_node)
        copy_click_probabilities = self.click_probabilities.copy()

        while len(self.queue)>0:
            self.actual_node = self.queue[0]
            if(np.random.normal(mean, std) > self.actual_node.price):
                copy_click_probabilities[:, self.actual_node.number] = 0
                self.activations[self.actual_node.number] += 1
                prob_click_a = copy_click_probabilities[self.actual_node.number][self.actual_node.secondary_a.number]
                prob_click_b = self.lambda_prob * copy_click_probabilities[self.actual_node.number][self.actual_node.secondary_b.number]
                v = np.random.uniform(.0, 1.)
                if(v < prob_click_a):
                    self.queue.append(self.actual_node.secondary_a)
                v = np.random.uniform(.0, 1.)
                if(v < prob_click_b):
                    self.queue.append(self.actual_node.secondary_b)
            self.queue = self.queue[1:]

    def simulate(self, starting_product_number=1, spin=10000):
        self.set_starting(product_number=starting_product_number)
        for i in range(spin):
            self.simulate_round()
            self.set_starting(product_number=starting_product_number)

    def get_results(self):
        return np.array(self.activations)/self.rounds

        
    def __str__(self):
        out = ""
        for i in range(len(self.products)):
            out += "Probability buy product P"+str(i)+" starting from P"+str(self.starting.number)+": "+str(self.activations[i]/self.rounds)+"\n"
        return out