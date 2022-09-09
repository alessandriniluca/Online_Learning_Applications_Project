import numpy as np

class QuantitiesEstimatorByFeatures:

    def __init__(self, n_products, n_features):
        self.sold_quantities = np.zeros((n_products, n_features))
        self.user_number = np.zeros((n_products, n_features))
        self.n_products = n_products

    def add_quantity(self, product, feature, quantity):
        feature = self.translate_feature(feature)
        self.sold_quantities[product][feature] += quantity
        self.user_number[product][feature] += 1


    def get_quantities(self, features):

        users = self.user_number.copy()
        buys = self.sold_quantities.copy()
        
        buys_context = np.zeros((self.n_products, 1))
        user_context = np.zeros((self.n_products, 1))

        features = self.translate_feature_list(features)

        for feature in features:
            buys_context[:, 0] += buys[:, feature]
            user_context[:, 0] += users[:, feature]

        user_context[user_context == 0] = 1

        res = buys_context/user_context
        return np.expand_dims(np.array(res[:, 0]), axis=0)


    def get_quantities_split(self, features):

            users = self.user_number.copy()
            buys = self.sold_quantities.copy()
            
            buys_context = np.zeros((self.n_products, 1))
            user_context = np.zeros((self.n_products, 1))

            features = self.translate_feature_list(features)

            for feature in features:
                buys_context[:, 0] += buys[:, feature]
                user_context[:, 0] += users[:, feature]

            user_context[user_context == 0] = 1
            buys_context[buys_context<50] = 0

            res = buys_context/user_context
            return np.array(res[:, 0])
            return np.expand_dims(np.array(res[:, 0]), axis=0)

    def get_quantities_divided(self, features):
        users = self.user_number.copy()
        buys = self.sold_quantities.copy()

        outcome = np.zeros((len(features), self.n_products))
        
        buys_context = np.zeros((self.n_products, 1))
        user_context = np.zeros((self.n_products, 1))

        features = self.translate_feature_list(features)

        for i, feature in enumerate(features):
            buys_context[:, 0] = buys[:, feature]
            user_context[:, 0] = users[:, feature]

            buys_context[buys_context < 50] = 0
            user_context[user_context == 0] = 1

            res = buys_context/user_context

            outcome[i, :] = np.array(res[:, 0])

        return outcome



    
    def translate_feature(self, feature):
        if feature == (0,0):
            feature = 0
        if feature == (0,1):
            feature = 1
        if feature == (1,0):
            feature = 2
        if feature == (1,1):
            feature = 3
        return feature

    def translate_feature_list(self, list):
        res = []

        for el in list:
            res.append(self.translate_feature(el))
        return res