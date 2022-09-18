import numpy as np

from optimizer.optimizer import Optimizer


def compute_alphas_prime(total_budget,
                         resolution,
                         products,
                         average_users_number,
                         basic_alphas,
                         alphas_functions,
                         one_per_product):
    # Compute all alpha primes: the new alpha ratio that I will have if a budget was allocated
    # Note that we will get expected value of dirichlet variables that are used to sample alphas
    alphas_prime = np.zeros((int(total_budget / resolution) + 1, len(products), 3))
    # for each budget allocation
    for budget_index, single_budget in enumerate(range(0, total_budget + resolution, resolution)):
        # for each product
        for product_index in range(len(products)):
            # for each class of user
            for class_index, users_of_current_class in enumerate(average_users_number):
                # set budget to corresponding product (using array with zeros for alpha function compatibility)
                # allocate just for one product the budget
                budgets = np.zeros(len(products))

                # if I have only one campaign per product (data are aggregate)
                # my budget will split between different class of users proportionally to the number of users
                # of the corresponding class
                if one_per_product:
                    budgets[product_index] = single_budget / sum(average_users_number) * users_of_current_class
                else:
                    budgets[product_index] = single_budget

                # compute deltas of weights

                delta_alpha_weights = alphas_functions[class_index](budgets)

                expected_new_weight = basic_alphas[class_index][product_index] + delta_alpha_weights[product_index]
                expected_new_alpha = expected_new_weight / sum(basic_alphas[class_index])

                alphas_prime[budget_index][product_index][class_index] = expected_new_alpha

    return alphas_prime


class FullOptimizer(Optimizer):
    """
    Basic class that implement a dynamic programming based budget optimizer

     Args:
        users_number (list): number of users per category
        min_budget (list): minimum budget constraints per campaign
        max_budget (list): maximum budget constraints per campaign
        buy_probs (np.array): buy probability matrix
        products (list): list of products
        total_budget (int): total budget for all campaigns
        resolution (int): minimum spendable amount
        alphas (np.array): contain alpha prime ratios (alpha that I will have given a certain budget allocated)
    """

    def __init__(self,
                 users_number,
                 min_budget,
                 max_budget,
                 total_budget,
                 resolution,
                 products,
                 mean_quantities,
                 buy_probs,
                 basic_alphas,
                 alphas_functions,
                 one_campaign_per_product=False):
        super().__init__(users_number,
                         min_budget,
                         max_budget,
                         total_budget,
                         resolution,
                         products,
                         mean_quantities,
                         buy_probs,
                         one_campaign_per_product=one_campaign_per_product)

        self.basic_alphas = basic_alphas
        self.alphas_functions = alphas_functions

        self.alphas = compute_alphas_prime(total_budget,
                                           resolution,
                                           products,
                                           users_number,
                                           basic_alphas,
                                           alphas_functions,
                                           one_campaign_per_product)
        print(self.alphas)

    @property
    def one_campaign_per_product(self):
        return self._one_campaign_per_product

    # one_campaign_per_product setter function
    @one_campaign_per_product.setter
    def one_campaign_per_product(self, new_value):
        # in order to improve efficiency
        if self._one_campaign_per_product != new_value:
            self._one_campaign_per_product = new_value
            self.alphas = compute_alphas_prime(self.total_budget,
                                               self.resolution,
                                               self.products,
                                               self.users_number,
                                               self.basic_alphas,
                                               self.alphas_functions,
                                               self.one_campaign_per_product)
