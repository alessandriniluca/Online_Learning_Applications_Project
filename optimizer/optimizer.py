import numpy as np


class Optimizer:

    def __init__(self, user_number, min_budget, max_budget, total_budget, resolution, prices, mean_quantities, alphas, buy_prob):
        self.user_number = user_number
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.total_budget = total_budget
        self.resolution = resolution
        self.prices = prices
        self.mean_quantities = mean_quantities
        self.alphas = alphas
        self.rows_income_per_budget = np.zeros((len(self.prices), int(max(max_budget) / resolution) + 1))
        self.final_table = np.zeros((len(self.prices) + 1, int(max(max_budget) / resolution) + 1))
        self.partition = []
        self.buy_prob = buy_prob

    def compute_rows(self):
        # for each user class
        for class_index, users_of_current_class in enumerate(self.user_number):
            # for each possible budget allocation
            for budget_index in range(int(max(self.total_budget) / self.resolution) + 1):
                # for each campaign
                for starting_prod in range(len(self.prices)):
                    # compute expected income
                    expected_income_per_user = 0
                    for arrival_prod in range(len(self.prices)):
                        expected_income_per_product = self.buy_prob[class_index][starting_prod][arrival_prod] * self.prices[arrival_prod]
                        # if data are aggregate
                        if self.mean_quantities.shape[0] == 1:
                            expected_income_per_user += expected_income_per_product * self.mean_quantities[0][arrival_prod]
                        # else consider different quantity means of different classes
                        else:
                            expected_income_per_user += expected_income_per_product * self.mean_quantities[class_index][arrival_prod]

                    # Check if we are estimating an aggregate, and thus we have just 5 alphas,
                    # or disaggregate, with different classes and alphas
                    if self.alphas.shape[2] == 1:
                        self.rows_income_per_budget[starting_prod][budget_index] += int(
                            self.alphas[budget_index][starting_prod][0] * users_of_current_class) * expected_income_per_user
                    else:
                        self.rows_income_per_budget[starting_prod][budget_index] += int(
                            self.alphas[budget_index][starting_prod][class_index] * users_of_current_class) * expected_income_per_user

    def build_final_table(self):
        for row in range(1, len(self.prices) + 1):
            actual_row = self.rows_income_per_budget[row - 1]
            previous_row = self.final_table[row - 1]

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
        final_budget = [0, 0, 0, 0, 0]
        budget_values = np.linspace(0, int(self.total_budget),
                                    num=int(self.total_budget / self.resolution) + 1)


        # Optimal overall budget allocation index
        start = np.argmax(self.final_table[-1] - budget_values)

        row_lenght = int(self.total_budget / self.resolution) + 1

        # Unroll the process to get budget allocations
        for i in range(len(self.prices) - 1, -1, -1):
            idx = i * row_lenght + start
            act = self.partition[idx][1]
            prv = self.partition[idx][0]
            final_budget[i] = budget_values[act]
            start = prv

        print("Expected profit:", np.max(self.final_table[-1] - budget_values))
        print("Budget allocation:", final_budget)
        return final_budget
