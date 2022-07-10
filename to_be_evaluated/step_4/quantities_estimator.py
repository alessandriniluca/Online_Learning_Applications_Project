import numpy as np


class QuantitiesEstimator:
    def __init__(self, products):
        self.sold_quantities = np.zeros(len(products))
        self.users_per_product = np.zeros(len(products))

    def update_quantities(self, rewards):
        for user in rewards:
            for product, quantity in user.bought_product:
                self.sold_quantities[product.number] += quantity
                self.users_per_product[product.number] += 1

    def get_quantities(self):
        quantities = []
        for qta, user in zip(self.sold_quantities, self.users_per_product):
            if user != 0:
                quantities.append(qta/user)
            else:
                quantities.append(1)
        return np.expand_dims(np.array(quantities), axis=0)