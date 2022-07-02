import numpy as np


class Optimizer2:

    def __init__(self, user_number, min_budget, max_budget, resolution, mean_prices, mean_quantities, alphas, buy_prob):
        self.user_number = user_number
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.resolution = resolution
        self.mean_prices = mean_prices
        self.mean_quantities = mean_quantities
        self.alphas = alphas
        self.rows_income_per_budget = np.zeros((len(self.mean_prices), int(max(max_budget) / resolution) + 1))
        self.final_table = np.zeros((len(self.mean_prices) + 1, int(max(max_budget) / resolution) + 1))
        self.partition = []
        self.buy_prob = buy_prob

    def compute_rows(self):
        # print("Table clicks:", self.buy_prob)
        for i, number in enumerate(self.user_number):
            # print("\nNew class", i, "users number:", number)
            for j in range(int(max(self.max_budget) / self.resolution) + 1):
                for starting_prod in range(len(self.mean_prices)):
                    income = 0
                    for arrival_prod in range(len(self.mean_prices)):
                        if self.mean_quantities.shape[0] == 1:
                            income += self.buy_prob[i][starting_prod][arrival_prod] * self.mean_prices[arrival_prod] * \
                                      self.mean_quantities[0][arrival_prod]
                        else:
                            income += self.buy_prob[i][starting_prod][arrival_prod] * self.mean_prices[arrival_prod] * \
                                      self.mean_quantities[i][arrival_prod]

                        # print("starting:", starting_prod, "arrive:", arrival_prod, "Buy prob:", self.buy_prob[i][starting_prod][arrival_prod], "price:", self.mean_prices[arrival_prod], "qta:", self.mean_quantities[i][arrival_prod])
                    # print("Number users:", number, "alpha:", self.alphas[j][starting_prod][0], "prod_int:", int(self.alphas[j][starting_prod][0] * number))
                    # Check if we are estimating an aggragate, and thus we have just 5 alphas, or disaggregate, whith different classes and alphas
                    if self.alphas.shape[2] == 1:
                        self.rows_income_per_budget[starting_prod][j] += int(
                            self.alphas[j][starting_prod][0] * number) * income
                    else:
                        self.rows_income_per_budget[starting_prod][j] += int(
                            self.alphas[j][starting_prod][i] * number) * income

                    # print(int(self.alphas[j][starting_prod][0] * number) * income)

        # print(self.rows_income_per_budget)

    def build_final_table(self):
        for row in range(1, len(self.mean_prices) + 1):
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
        budget_values = np.linspace(int(self.min_budget[0]), int(self.max_budget[0]),
                                    num=int(self.max_budget[0] / self.resolution) + 1)
        # print(self.final_table)
        start = np.argmax(self.final_table[-1] - budget_values)
        # print(start)
        a = self.final_table[-1] - budget_values
        # print(max(a))
        row_lenght = int(self.max_budget[0] / self.resolution) + 1
        # print(row_lenght)
        for i in range(len(self.mean_prices) - 1, -1, -1):
            idx = i * row_lenght + start
            act = self.partition[idx][1]
            prv = self.partition[idx][0]
            # print(prv, act)
            final_budget[i] = budget_values[act]
            start = prv
        print("Optimizer earning:", np.max(self.final_table[-1] - budget_values))
        print("Budget:", final_budget)
        return final_budget
