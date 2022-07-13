import math

from common.utils import get_logger, get_products
from environment.product import Product
from environment.user import User
import numpy as np

logger = get_logger(__name__)


class Environment:
    """
    This class defines the environment in which simulations will be performed
    """

    def __init__(self, configuration, alphas_functions):
        self.configuration_ = """
        Args:
            configuration (Configuration):
                current environment configuration
            alphas_functions (list):
                set of alpha functions
        """

        self.configuration = configuration

        # Create alpha functions with given parameters
        self.alphas_functions = alphas_functions

        # instantiate products from parameters
        self.products = get_products(configuration.products_parameters)

        # To keep track of number of users at each round
        self.users_per_round = []

        logger.info("Environment initialized")

    def round(self, budget):
        """
        Simulate a round given a budget allocation
        """
        # Log the current round number
        logger.info("[Round " + str(len(self.users_per_round)) + "] started")

        # sample the total number of users for the problem
        n_users = np.random.normal(self.configuration.average_users_number,
                                   self.configuration.std_users)

        logger.debug("TOTAL users in this round: " + str(math.ceil(sum(n_users))))

        # Compute alpha increments
        delta_increment = []
        for function in self.alphas_functions:
            delta_increment.append(np.concatenate((function(budget), np.array([0]))))

        logger.debug("Delta increments: " + str(delta_increment))

        # Compute new weights
        new_weights = self.configuration.basic_alphas + np.array(delta_increment)

        # Now make sure that the previous sum of weights is the same of the new one
        # Subtracting increment from alpha_zero
        for class_index in range(len(new_weights)):
            if sum(delta_increment[class_index]) > new_weights[class_index][5]:
                raise ValueError("Delta increment exceeding alpha zero weights")
            new_weights[class_index][5] -= sum(delta_increment[class_index])

        # Sample alphas for this round from dirichlet distributions
        actual_alpha = np.array([])
        for i in range(len(new_weights)):
            actual_alpha = np.concatenate(
                (actual_alpha, np.random.dirichlet(20 * new_weights[i] / sum(new_weights[i]))), axis=0)
        actual_alpha = actual_alpha.reshape(3, 6)

        logger.debug("Alpha ratios: " + str(actual_alpha))

        # Apply alpha ratios to all users
        users_per_category = (actual_alpha * n_users[:, np.newaxis]).astype(int)
        total_number_users = sum(users_per_category)
        logger.debug("Users per category " + str(users_per_category))
        logger.debug("Sum: " + str(sum(total_number_users)))

        # discard alpha_0, user that visit competitor's website
        users_per_category = (users_per_category[:, :5])

        # instantiate all users for this round
        this_round_users = []
        for class_index in range(len(users_per_category)):
            for start_product_index in range(len(users_per_category[class_index])):
                for _ in range(users_per_category[class_index][start_product_index]):

                    # instantiate user of class (class_index) that starts from product (primary_product_index)
                    user = User(
                        self.configuration.reservation_price_means[class_index],
                        self.configuration.reservation_price_std_dev[class_index],
                        self.configuration.quantity_means[class_index],
                        self.configuration.quantity_std_dev[class_index],
                        user_class=class_index,
                        # Note that we will have an effective split on feature 0
                        # The split of feature 1 instead will be performed only after seeing feature 0 == 0
                        features=[0, 0] if class_index == 0 else (
                            [0, 1] if class_index == 1 else ([1, 1] if np.random.uniform(0, 1) < 0.5 else [1, 0])),
                        starting_product=start_product_index,
                        # Make a copy of the graph for each user, since exploration will change graph probabilities
                        graph_clicks=self.configuration.graph_clicks.copy()
                    )

                    this_round_users.append(user)

                    starting_prod = self.products[start_product_index]

                    # Keep track of "open tabs" aka products to visit
                    product_queue = [starting_prod]

                    # While we have open tab
                    while len(product_queue) > 0:
                        # Pick the first tab in the queue
                        prod = product_queue[0]
                        user.add_seen_product(prod)

                        # check if user is willing to buy product
                        if user.has_bought(prod):

                            # handle the quantity bought
                            qta = user.quantity_bought(prod)
                            user.add_bought_product(prod, qta)

                            # if click on 1st secondary
                            if user.product_click(prod, prod.secondary_a):
                                product_queue.append(prod.secondary_a)

                            # if click on 2nd secondary
                            if np.random.uniform(0.0, 1.0) < self.configuration.lambda_prob:
                                if user.product_click(prod, prod.secondary_b):
                                    product_queue.append(prod.secondary_b)

                        # remove the product from the queue
                        product_queue = product_queue[1:]

        # update history
        self.users_per_round.append(this_round_users)

        return this_round_users, sum(total_number_users)
