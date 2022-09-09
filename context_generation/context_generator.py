
import numpy as np
from context_generation.context import Context
from environment.environment_context import Environment
from optimizer.estimator import Estimator
from optimizer.optimizer_context import Optimizer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ConstantKernel as C
from common.utils import load_static_env_configuration, load_static_sim_configuration, get_test_alphas_functions, \
    LearnerType, translate_feature_group
from probability_calculator.quantities_estimator_by_feature import QuantitiesEstimatorByFeatures
from matplotlib import pyplot as plt



class ContextGenerator:

    def __init__(self, n_arms, arms, learner_type):
        self.active_contexts = []
        self.n_arms = n_arms
        self.arms = arms
        self.learner_type = learner_type
        self.splitted_features = []
        self.average_users_per_feature = [50, 50, 25, 25]
        self.n_learners=5
        self.quantity_estimator = QuantitiesEstimatorByFeatures(5, 4)

        # self.rewards_per_feature = np.zeros((self.n_arms, self.n_learners), dtype=np.ndarray)
        self.rewards_per_feature = []

        # for i in range(self.n_arms):
        #     for j in range(self.n_learners):
        #         #rewards_per_product = [[[], []], [[], []]]
        #         self.rewards_per_feature[i][j] = [[[], []], [[], []]]


    def start(self):
        context = Context(self.average_users_per_feature, self.quantity_estimator, self.rewards_per_feature, self.n_arms, self.arms, self.learner_type, n_learners=5, features=[(0,0), (0,1), (1,0), (1,1)])
        self.active_contexts.append(context)
        self.splitted_features.append([(0,0), (0,1), (1,0), (1,1)])


    def add_reward(self, product_number, arm_idx, user_feature, reward):
        feature1 = user_feature[0]
        feature2 = user_feature[1]
        float_arm = None
        for s in self.splitted_features:
            if (feature1, feature2) in s:
                # print("-----", self.arms[arm_idx], self.get_users_in_context(s), self.get_users_in_context([user_feature]))
                float_arm = self.arms[arm_idx]/self.get_users_in_context(s) * self.get_users_in_context([user_feature])
        self.rewards_per_feature.append([reward, product_number, float_arm, user_feature])

    def get_users_in_context(self, feature_list):
        tot = 0
        for feature in self.translate_features(feature_list):
            tot += self.average_users_per_feature[feature]
        return tot

        
    
    def split(self):
        self.active_contexts = []
        self.splitted_features = []

        no_split_alphas = self.get_alphas_by_feature([(0,0), (0,1), (1,0), (1,1)])
        print( "PPPPP", no_split_alphas.shape)

        reward_no_split = self.optimize([no_split_alphas], [[(0,0), (0,1), (1,0), (1,1)]])

        # try split first feature
        first_feature_1 = self.get_alphas_by_feature([(0,0), (0,1)])
        first_feature_2 = self.get_alphas_by_feature([(1,0), (1,1)])
        
        # a = self.optimize(first_feature_1, [(0,0), (0,1)]) 
        # b = self.optimize(first_feature_2, [(1,0), (1,1)])
        # print("---a:", a, "--b:", b)
        reward_first_feature = self.optimize([first_feature_1, first_feature_2], [[(0,0), (0,1)], [(1,0), (1,1)]])


        #try split second feature
        second_feature_1 = self.get_alphas_by_feature([(0,0), (1,0)])
        second_feature_2 = self.get_alphas_by_feature([(0,1), (1,1)])
        # a = self.optimize(second_feature_1, [(0,0), (1,0)]) 
        # b = self.optimize(second_feature_2, [(0,1), (1,1)])
        # print("22---a:", a, "--b:", b)
        reward_second_feature = self.optimize([second_feature_1, second_feature_2], [[(0,0), (1,0)], [(0,1), (1,1)]])

        print("------ reward no split", reward_no_split, "Reward first feature", reward_first_feature, "Reward second feature", reward_second_feature)
        if reward_no_split > reward_first_feature and reward_no_split > reward_second_feature:
            context = Context(self.average_users_per_feature, self.quantity_estimator, self.rewards_per_feature, self.n_arms, self.arms, self.learner_type, n_learners=5, features=[(0,0), (0,1), (1,0), (1,1)])
            self.active_contexts.append(context)
            self.splitted_features.append([(0,0), (0,1), (1,0), (1,1)])

        elif reward_first_feature > reward_second_feature:
            
            # provo ulteriore split su feature 2
            aggregate_1 = self.optimize([first_feature_1], [[(0,0), (0,1)]]) 

            set1 = self.get_alphas_by_feature([(0,0)])
            set2 = self.get_alphas_by_feature([(0,1)])
            splitted_reward = self.optimize([set1, set2], [[(0,0)], [(0,1)]])

            print("------1st aggregate_1", aggregate_1, "splitted_reward", splitted_reward)

            
            if aggregate_1 < splitted_reward:
                # split primo dei due gruppi
                context = Context(self.average_users_per_feature, self.quantity_estimator, self.rewards_per_feature, self.n_arms, self.arms, self.learner_type, n_learners=5, features=[(0,0)])
                self.active_contexts.append(context)
                context = Context(self.average_users_per_feature, self.quantity_estimator, self.rewards_per_feature, self.n_arms, self.arms, self.learner_type, n_learners=5, features=[(0,1)])
                self.active_contexts.append(context)
                self.splitted_features.append([(0,0)])
                self.splitted_features.append([(0,1)])
            else:
                context = Context(self.average_users_per_feature, self.quantity_estimator, self.rewards_per_feature, self.n_arms, self.arms, self.learner_type, n_learners=5, features=[(0,0), (0,1)])
                self.active_contexts.append(context)
                self.splitted_features.append([(0,0), (0,1)])

            aggregate_2 = self.optimize([first_feature_2], [[(1,0), (1,1)]])
            
            set1 = self.get_alphas_by_feature([(1,0)])
            set2 = self.get_alphas_by_feature([(1,1)])
            splitted_reward = self.optimize([set1, set2], [[(1,0)], [(1,1)]])

            print("------1st aggregate_2", aggregate_2, "splitted_reward", splitted_reward)

            if aggregate_2 < splitted_reward:
                context = Context(self.average_users_per_feature, self.quantity_estimator, self.rewards_per_feature, self.n_arms, self.arms, self.learner_type, n_learners=5, features=[(1,0)])
                self.active_contexts.append(context)
                context = Context(self.average_users_per_feature, self.quantity_estimator, self.rewards_per_feature, self.n_arms, self.arms, self.learner_type, n_learners=5, features=[(1,1)])
                self.active_contexts.append(context)
                self.splitted_features.append([(1,0)])
                self.splitted_features.append([(1,1)])
            else:
                context = Context(self.average_users_per_feature, self.quantity_estimator, self.rewards_per_feature, self.n_arms, self.arms, self.learner_type, n_learners=5, features=[(1,0), (1,1)])
                self.active_contexts.append(context)
                self.splitted_features.append([(1,0), (1,1)])
            

        else:         
            # provo ulteriore split su feature 2
            aggregate_1 = self.optimize([second_feature_1], [[(0,0), (1,0)]])

            set1 = self.get_alphas_by_feature([(0,0)])
            set2 = self.get_alphas_by_feature([(1,0)])
            splitted_reward = self.optimize([set1, set2], [[(0,0)], [(1,0)]])
            print("------2nd aggregate_1", aggregate_1, "splitted_reward", splitted_reward)

            if aggregate_1 < splitted_reward:
                # split primo dei due gruppi
                context = Context(self.average_users_per_feature, self.quantity_estimator, self.rewards_per_feature, self.n_arms, self.arms, self.learner_type, n_learners=5, features=[(0,0)])
                self.active_contexts.append(context)
                context = Context(self.average_users_per_feature, self.quantity_estimator, self.rewards_per_feature, self.n_arms, self.arms, self.learner_type, n_learners=5, features=[(1,0)])
                self.active_contexts.append(context)
                self.splitted_features.append([(0,0)])
                self.splitted_features.append([(1,0)])
            else:
                context = Context(self.average_users_per_feature, self.quantity_estimator, self.rewards_per_feature, self.n_arms, self.arms, self.learner_type, n_learners=5, features=[(0,0), (1,0)])
                self.active_contexts.append(context)
                self.splitted_features.append([(0,0),(1,0)])

            aggregate_2 = self.optimize([second_feature_2], [[(0,1), (1,1)]])

            set1 = self.get_alphas_by_feature([(0,1)])
            set2 = self.get_alphas_by_feature([(1,1)])
            splitted_reward = self.optimize([set1, set2], [[(0,1)], [(1,1)]])
            print("------2nd aggregate_2", aggregate_2, "splitted_reward", splitted_reward)

            if aggregate_2 < splitted_reward:
                # split primo dei due gruppi
                context = Context(self.average_users_per_feature, self.quantity_estimator, self.rewards_per_feature, self.n_arms, self.arms, self.learner_type, n_learners=5, features=[(0,1)])
                self.active_contexts.append(context)
                context = Context(self.average_users_per_feature, self.quantity_estimator, self.rewards_per_feature, self.n_arms, self.arms, self.learner_type, n_learners=5, features=[(1,1)])
                self.active_contexts.append(context)
                self.splitted_features.append([(0,1)])
                self.splitted_features.append([(1,1)])
            else:
                context = Context(self.average_users_per_feature, self.quantity_estimator, self.rewards_per_feature, self.n_arms, self.arms, self.learner_type, n_learners=5, features=[(0,1), (1,1)])
                self.active_contexts.append(context)
                self.splitted_features.append([(0,1), (1,1)])


    def optimize(self, alphas, features):

        # print("Tipo alpha:", type(alphas), alphas)
        env_configuration = load_static_env_configuration("configurations/environment/static_conf_1.json")
        sim_configuration = load_static_sim_configuration("configurations/simulation/sim_conf_1.json")
        alphas_functions = get_test_alphas_functions()

        env = Environment(
            configuration=env_configuration,
            alphas_functions=alphas_functions
        )
    
        estimator = Estimator(env.configuration.graph_clicks,
                        env.products,
                        env.configuration.lambda_prob,
                        env.configuration.reservation_price_means,
                        env.configuration.reservation_price_std_dev
                        )

        buy_probs = estimator.get_buy_probs()


        qtas = []
        for features_group in features:
            qta = np.zeros((len(features_group), 5))
            for i, f in enumerate(features_group):
                qta[i, :] = self.quantity_estimator.get_quantities_split(f)
            qtas.append(qta)

        real_alphas = np.zeros((self.n_arms, self.n_learners, len(features)))
        for i in range(len(alphas)):
            real_alphas[:, :, i] = (alphas[i])[:, :, 0]


        # print("TRADUZIONE", self.translate_features(features))
        optimizer = Optimizer(
            users_number=env.configuration.average_users_number,
            min_budget=sim_configuration["min_budget"],
            max_budget=sim_configuration["max_budget"],
            total_budget=sim_configuration["total_budget"],
            resolution=sim_configuration["resolution"],
            products=env.products,
            mean_quantities=qtas,#self.quantity_estimator.get_quantities(features),
            buy_probs=buy_probs,
            alphas=real_alphas,
            features_division=translate_feature_group(features),#[self.translate_features(features)],
            one_campaign_per_product=False,
            multiple_quantities=True
        )
        
        # print(" ---- ", self.quantity_estimator.get_quantities_divided(features))

        optimizer.run_optimization()
        current_allocation, expected_profit = optimizer.find_best_allocation()
        return expected_profit #* total_users_in_context / (total_users_in_context + users_to_divide)


    def update(self, pulled_arms, rewards, user_features):
        for i, context in enumerate(self.active_contexts):
            context.update(pulled_arms[i*5:i*5+5], rewards, user_features)
        for i in range(self.n_learners):
            for r, f in zip(rewards[i], user_features):
                self.add_reward(i, pulled_arms[i], f, r)


    def update_quantities(self, rewards):
        for user in rewards:
            for product, quantity in user.bought_product:
                self.quantity_estimator.add_quantity(product.number, user.features, quantity)


    def get_alphas_by_feature(self, features):
        # print("------ FETAURES:", features)
        total_users = 0
        gps = []
        alphas = np.zeros((self.n_arms, self.n_learners, 1))
        """

        Args:
            features (list): it is a list like [(0,0), (1,0)] of the feature you are interested in
        """

        alpha = .5
        kernel = C(5, constant_value_bounds="fixed") * RBF(20, length_scale_bounds="fixed")

        for i in range(self.n_learners):
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=alpha ** 2,
                normalize_y=True,
                n_restarts_optimizer=9
            )

            gps.append(gp)


            x = []
            y = []

            for user in self.rewards_per_feature:
                if user[3] in features and user[1] == i:
                    x.append(user[2] * self.get_users_in_context(features) / self.get_users_in_context([user[3]]))
                    y.append(user[0])
                    total_users += 1

            x = np.atleast_2d(x).T

            # Retrain the GP
            gp.fit(x, y)


            mean, sigma = gp.predict(np.atleast_2d(self.arms).T, return_std=True)
            lower_bound = mean-sigma

            alphas[:, i, 0] = np.array(lower_bound)


            # x_pred = np.atleast_2d(self.arms).T
            # y_pred, sigma = gp.predict(x_pred, return_std=True)
            # plt.figure("GP: "+str(i)+" FEATURES: "+str(features))
            # plt.ylim(-0.5, 1.5)
            # plt.title("GP product: "+str(i)+" FEATURES: "+str(features))
            # plt.plot(x.ravel(), y, 'ro', label=r'Observed Clicks')
            # plt.plot(x_pred, y_pred, 'b-', label=r'Predicted clicks')
            # plt.fill(
            #     np.concatenate([x_pred, x_pred[::-1]]), 
            #     np.concatenate([y_pred - 1.0* sigma, (y_pred + 1.0 * sigma)[::-1]]),
            #     alpha=.5, fc='b', ec='None', label='95% conf interval')
            # plt.xlabel('$x$')
            # plt.ylabel('$n(x)$')
            # plt.legend(loc='lower right')
            # plt.show()

        

        return alphas
            



    def lower_bound(self, delta, set_cardinality):
        # print("Lower bound:", delta, set_cardinality)
        if set_cardinality == 0:
            return 1000
        return np.sqrt(-np.log2(delta)/(2*set_cardinality))


    def translate_features(self, features):
        res = []
        for f in features:
            if f == (0,0):
                res.append(0)
            elif f == (0,1):
                res.append(1)
            elif f == (1,0):
                res.append(2)
            elif f == (1,1):
                res.append(3)
        return res


    def get_expected_rewards(self):
        alphas_prime = np.zeros((self.n_arms, self.n_learners, len(self.splitted_features)))
        for idx, context in enumerate(self.active_contexts):
            alphas_prime[:, :, idx] = context.get_expected_rewards()[:, :, 0]

        return self.splitted_features, alphas_prime

    
    def get_quantities(self):
        quantities = np.zeros((len(self.splitted_features), self.n_learners))
        for idx, context in enumerate(self.splitted_features):
            qta = self.quantity_estimator.get_quantities(self.splitted_features[idx])
            quantities[idx, :] = qta[0, :]

        return quantities

    def compute_mean(self, list):
        if len(list) == 0:
            return 0
        return np.mean(list)
