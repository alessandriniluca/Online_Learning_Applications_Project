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
                 features_division,
                 one_campaign_per_product=False,
                 multiple_quantities=False):
        self.users_number = [users_number[0], users_number[1], users_number[2]/2, users_number[2]/2]
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.total_budget = total_budget
        self.resolution = resolution
        self.products = products
        self.prices = get_prices(products)
        self.mean_quantities = mean_quantities
        self.buy_prob = buy_probs
        self._one_campaign_per_product = one_campaign_per_product
        self.features_division = features_division
        self.multiple_quantities=multiple_quantities

        self.alphas = alphas

        self.number_of_budgets_to_evaluate = int(total_budget / resolution) + 1

        # Note that if I want to exploit context I will need to have multiple campaigns
        self.number_of_campaigns = len(self.prices) * len(self.features_division)#len(self.prices) if one_campaign_per_product else (len(self.prices) * len(users_number))
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
        # print("----INIZIO----")
        # for each possible budget allocation
        for budget_index in range(int(self.total_budget / self.resolution) + 1):
            # for each campaign
            for campaign in range(self.number_of_campaigns):
                # Same starting product is encountered more times if more context are provided
                # And so there will be more campaigns for the same product
                starting_prod = int(campaign % len(self.prices))
                current_division_feature = int(campaign / len(self.prices))


                # compute expected income
                # print("FEATURE ORA: ", self.features_division[current_division_feature])
                for idx, feature in enumerate(self.features_division[current_division_feature]):
                    expected_income_per_user = 0
                    if feature > 2:
                        current_class = 2
                    else:
                        current_class = feature
                    # print(" -- considero:", feature, "class:", current_class)

                    # print("start prod:", starting_prod, "current_class:", current_class)

                    incomes = []
                    for arrival_prod in range(len(self.prices)):
                        # We can exploit the context, the expected income will be the one of the corrisponding class
                        expected_income_per_product = self.buy_prob[current_class][starting_prod][arrival_prod] * \
                                                    self.prices[arrival_prod]
                        incomes.append(expected_income_per_product)

                        if self.multiple_quantities:
                        # Same for the expected income per user
                            expected_income_per_user += expected_income_per_product * (self.mean_quantities[current_division_feature])[idx][arrival_prod]
                        else:
                            expected_income_per_user += expected_income_per_product * self.mean_quantities[current_division_feature][arrival_prod]

                        # print("EXp:", expected_income_per_user, "counter:", current_division_feature)
                    # print("alpha:", self.alphas[budget_index][starting_prod][current_division_feature], "users:", self.users_number[current_class] , "income per user", expected_income_per_user)
                    # print("current class:", current_class, "income per product:", incomes)
                    # print("UTENTI:", self.users_number[current_class])
                    self.rows_income_per_budget[campaign][budget_index] += (
                        self.alphas[budget_index][starting_prod][current_division_feature] * self.users_number[current_class]) * expected_income_per_user
                    # print("------------- IDX:", current_division_feature, "FEATURE", feature, "PP", self.rows_income_per_budget[campaign][budget_index])

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
                max_val = -np.inf
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
            #print(self.rows_income_per_budget)
        self.build_final_table()

    def reset(self):
        self.number_of_campaigns = len(self.prices) * len(self.features_division)
        self.rows_income_per_budget = np.zeros((self.number_of_campaigns, self.number_of_budgets_to_evaluate))
        self.final_table = np.zeros((self.number_of_campaigns + 1, self.number_of_budgets_to_evaluate))
        self.partition = []

    def set_alphas(self, alphas):
        self.alphas = alphas

    def set_buy_probabilities(self, buy_prob):
        """Method needed to perform step 5 where buy probabilites change

        Args:
            buy_prob (matrix): estimated buy prob
        """
        self.buy_prob = buy_prob
