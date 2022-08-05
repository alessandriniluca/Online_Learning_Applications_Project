import numpy as np

from common.utils import get_prices


class Optimizer:
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
                 alphas,
                 one_campaign_per_product=False):
        self.users_number = users_number
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.total_budget = total_budget
        self.resolution = resolution
        self.products = products
        self.prices = get_prices(products)
        self.mean_quantities = mean_quantities
        self.buy_prob = buy_probs
        self._one_campaign_per_product = one_campaign_per_product

        self.alphas = alphas

        self.number_of_budgets_to_evaluate = int(total_budget / resolution) + 1

        # Note that if I want to exploit context I will need to have multiple campaigns
        self.number_of_campaigns = len(self.prices) if one_campaign_per_product else (len(self.prices) * len(users_number))

        self.rows_income_per_budget = np.zeros((self.number_of_campaigns, self.number_of_budgets_to_evaluate))
        self.final_table = np.zeros((self.number_of_campaigns + 1, self.number_of_budgets_to_evaluate))
        self.partition = []

    @property
    def one_campaign_per_product(self):
        return self._one_campaign_per_product

    # one_campaign_per_product setter function
    @one_campaign_per_product.setter
    def one_campaign_per_product(self, new_value):
        self._one_campaign_per_product = new_value

    def compute_rows(self):
        # for each possible budget allocation
        for budget_index in range(int(self.total_budget / self.resolution) + 1):
            # for each campaign
            for campaign in range(self.number_of_campaigns):
                # Same starting product is encountered more times if more context are provided
                # And so there will be more campaigns for the same product
                starting_prod = int(campaign % len(self.prices))
                current_class = int(campaign / len(self.prices))
                # compute expected income
                expected_income_per_user = 0
                for arrival_prod in range(len(self.prices)):
                    # We can exploit the context, the expected income will be the one of the corrisponding class
                    expected_income_per_product = self.buy_prob[current_class][starting_prod][arrival_prod] * \
                                                  self.prices[arrival_prod]

                    # Same for the expected income per user
                    expected_income_per_user += expected_income_per_product * self.mean_quantities[current_class][
                        arrival_prod]

                self.rows_income_per_budget[campaign][budget_index] = int(
                    self.alphas[budget_index][starting_prod][
                        current_class] * self.users_number[current_class]) * expected_income_per_user

    def compute_rows_aggregate_campaigns(self):
        # for each user class
        for class_index, users_of_current_class in enumerate(self.users_number):
            # for each possible budget allocation
            for budget_index in range(int(self.total_budget / self.resolution) + 1):
                # for each campaign
                for starting_prod in range(len(self.prices)):
                    # compute expected income
                    expected_income_per_user = 0
                    for arrival_prod in range(len(self.prices)):
                        expected_income_per_product = self.buy_prob[class_index][starting_prod][arrival_prod] * \
                                                      self.prices[arrival_prod]
                        # if data are aggregate TODO
                        if self.mean_quantities.shape[0] == 1:
                            expected_income_per_user += expected_income_per_product * self.mean_quantities[0][
                                arrival_prod]
                        # else consider different quantity means of different classes
                        else:
                            expected_income_per_user += expected_income_per_product * \
                                                        self.mean_quantities[class_index][arrival_prod]

                    # Check if we are estimating an aggregate, and thus we have just 5 alphas,
                    # or disaggregate, with different classes and alphas
                    if self.alphas.shape[2] == 1:
                        self.rows_income_per_budget[starting_prod][budget_index] += int(
                            self.alphas[budget_index][starting_prod][
                                0] * users_of_current_class) * expected_income_per_user
                    else:
                        self.rows_income_per_budget[starting_prod][budget_index] += int(
                            self.alphas[budget_index][starting_prod][
                                class_index] * users_of_current_class) * expected_income_per_user

    def build_final_table(self):
        for row in range(1, self.number_of_campaigns + 1):
            actual_row = self.rows_income_per_budget[row - 1]
            previous_row = self.final_table[row - 1]

            # initialize prev and act to prevent compiler error
            prev = None
            act = None

            for i in range(0, len(actual_row)):
                max_val = -1000
                for j in range(0, i + 1):
                    if actual_row[i - j] + previous_row[j] > max_val:
                        max_val = actual_row[i - j] + previous_row[j]
                        prev = j
                        act = i - j
                self.partition.append(np.array([prev, act]))
                self.final_table[row][i] = max_val

    def find_best_allocation(self):
        final_budget = np.zeros(self.number_of_campaigns)
        budget_values = np.linspace(0, int(self.total_budget), num=self.number_of_budgets_to_evaluate)

        # Optimal overall budget allocation index
        start = np.argmax(self.final_table[-1] - budget_values)

        # Unroll the process to get budget allocations
        for i in range(self.number_of_campaigns - 1, -1, -1):
            idx = i * self.number_of_budgets_to_evaluate + start
            act = self.partition[idx][1]
            prv = self.partition[idx][0]
            if act is None:
                # TODO needs to be further investigate...
                raise ValueError("act is none")
            if prv is None:
                raise ValueError("prv is none")
            final_budget[i] = budget_values[act]
            start = prv

        expected_profit = np.max(self.final_table[-1] - budget_values)

        # print("Expected profit:", expected_profit)
        # print("Budget allocation:", final_budget)
        return final_budget, expected_profit

    def run_optimization(self):
        self.reset()
        if self.one_campaign_per_product:
            self.compute_rows_aggregate_campaigns()
        else:
            self.compute_rows()
        self.build_final_table()

    def reset(self):
        self.number_of_campaigns = len(self.prices) if self.one_campaign_per_product else (
                    len(self.prices) * len(self.users_number))
        self.rows_income_per_budget = np.zeros((self.number_of_campaigns, self.number_of_budgets_to_evaluate))
        self.final_table = np.zeros((self.number_of_campaigns + 1, self.number_of_budgets_to_evaluate))
        self.partition = []

    def set_alphas(self, alphas):
        self.alphas = alphas
