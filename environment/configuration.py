import numpy as np


class Configuration:
    """
    Basic class for define a static configuration of the environment
    """

    def __init__(self,
                 average_users_number,
                 std_users,
                 basic_alphas,
                 products_parameters,
                 lambda_prob,
                 graph_clicks,
                 reservation_price_means,
                 reservation_price_std_dev,
                 quantity_means,
                 quantity_std_dev
                 ):
        """
                Args:
                    average_users_number (list):
                        number of users per category
                    std_users (list):
                        standard deviations of the user number
                    basic_alphas (list):
                        array that contains starting weights of dirichlet distribution
                        that will provide values of alphas
                        note that we will have 6 weights for each class (18 weights in total)
                    products_parameters (list):
                        list of products parameters
                    lambda_prob (float):
                        probability to see the second suggested product in a page
                    graph_clicks (np.array):
                        The graph that will define click probabilities starting
                        from one node (product) to another
                    reservation_price_means (np.array):
                        reservation price for each product and for each class
                    reservation_price_std_dev (np.array):
                        reservation price std for each product and for each class
                    quantity_means (np.array):
                        quantity means for each product and for each class
                    quantity_std_dev (np.array):
                        quantity means std for each product and for each class

                """
        self.average_users_number = average_users_number
        self.std_users = std_users

        # always consider dirichlet weights
        self.basic_alphas = basic_alphas

        # lambda, given fixed by the problem
        self.lambda_prob = lambda_prob

        # products sold, 5 in this case
        self.products_parameters = products_parameters

        self.graph_clicks = np.array(graph_clicks)

        self.reservation_price_means = np.array(reservation_price_means)
        self.reservation_price_std_dev = np.array(reservation_price_std_dev)

        self.quantity_means = np.array(quantity_means)
        self.quantity_std_dev = np.array(quantity_std_dev)
