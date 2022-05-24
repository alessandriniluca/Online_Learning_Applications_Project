from environment.user import User
import numpy as np

class Environment:

    def __init__(self, average_users_number, std_users, basic_alphas, alphas_functions, products, lambda_prob, graph_clicks):
        # parameters to select the number of users of the specific round
        self.average_users_number = average_users_number
        self.std_users = std_users

        # value of alpha, that will be between basic_alphas and basic_alphas + max(alphas_functions)
        self.basic_alphas = basic_alphas
        self.alphas_functions = alphas_functions

        # labmda, given fixed by the problem
        self.lambda_prob = lambda_prob

        # products sold, 5 in this case
        self.products = products

        # click graph
        self.graph_clicks = graph_clicks

        self.reservation_price_means = np.array([[9, 14, 21, 8, 4], [10, 15, 22, 9, 2], [8, 12, 25, 10, 6]])
        self.reservation_price_std_dev = np.array([[2, 3, 2, 3, 3], [2, 1, 2, 1, 2], [4, 3, 2, 3, 2]])

        self.quantity_means = np.array([[2, 3, 3, 3, 1], [1, 1, 1, 2, 3], [3, 4, 5, 5,5]])
        self.quantity_std_dev = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])

        self.users_per_round = []

    def product_click(self, main_prod, secondary_prod):
        return np.random.uniform(0.0, 1.0) < self.graph_clicks[main_prod.number][secondary_prod.number]

    def update_graph(self, graph, prod):
        graph[:, prod.number] = 0
        return graph

    def round(self, budget):
        # sample the total number of users for the problem
        n_users = np.random.normal(self.average_users_number, self.std_users)
        res = self.basic_alphas + self.alphas_functions(budget)
        actual_alpha = np.array([])
        for i in range(len(res)):
            actual_alpha = np.concatenate((actual_alpha, np.random.dirichlet(res[i])), axis=0)                
        actual_alpha = actual_alpha.reshape(3, 6)

        # users_per_category = (n_users * actual_alpha).astype(int)
        users_per_category = (actual_alpha*n_users[:, np.newaxis]).astype(int)

        # discard alpha_0, user that visit competitor's website 
        users_per_category = (users_per_category[:, :5]) #.reshape(3, 5)
        this_round_users = []

        for i in range(len(users_per_category)):
            for j in range(len(users_per_category[i])):
                for _ in range(users_per_category[i][j]):
                    # print("Created user of class:", i, "starting with product:", j)
                    user = User(
                        self.reservation_price_means[i],
                        self.reservation_price_std_dev[i],
                        self.quantity_means[i],
                        self.quantity_std_dev[i],
                        user_class=i, 
                        starting_product = j
                    )

                    this_round_users.append(user)

                    graph_clicks = self.graph_clicks.copy()

                    starting_prod = self.products[j]
                    product_queue = []
                    product_queue.append(starting_prod)

                    while len(product_queue) > 0:
                        prod = product_queue[0]
                        user.add_seen_product(prod)
                        graph_clicks = self.update_graph(graph_clicks, prod)
                        # print(graph_clicks)

                        if user.has_bought(prod):
                            # print("\tbought product:", prod.number)
                            qta = user.quantity_bought(prod)
                            user.add_bought_product(prod, qta)

                            if self.product_click(prod, prod.secondary_a):
                                # print("\tclick secondary A:", prod.secondary_a.number)
                                product_queue.append(prod.secondary_a)
                            if np.random.uniform(0.0, 1.0) < self.lambda_prob:
                                if self.product_click(prod, prod.secondary_b):
                                    product_queue.append(prod.secondary_b)
                                    # print("\tclick secondary B:", prod.secondary_b.number)

                        product_queue = product_queue[1:]

        self.users_per_round.append(this_round_users)
        return this_round_users
                        


                    




