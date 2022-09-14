import numpy as np

from bandits.multi_learner import MultiLearner
from probability_calculator.quantities_estimator import QuantitiesEstimator

class Context:
    def unison_shuffled_copies(self, a, b):
            assert len(a) == len(b)
            p = np.random.permutation(len(a))
            return a[p], b[p]

    def __init__(self, average_users_per_feature, quantity_estimator, rewards_per_feature, n_arms, budgets, learner_type, n_learners, features):
        self.rewards_per_feature = rewards_per_feature
        self.n_arms = n_arms
        self.budgets = budgets
        self.learner_type = learner_type
        self.n_learners = n_learners
        self.features = features
        self.average_users_per_feature = average_users_per_feature
        self.quantity_estimator = quantity_estimator
        self.mab = MultiLearner(self.n_arms, self.budgets, self.learner_type, self.n_learners)


        # if len(self.rewards_per_feature) > 0:
        #     for arm in range(self.n_arms):
        #         rewards = []
        #         pulled_arms = []
        #         for i in range(self.n_learners):
        #             #pulled_arms.append(arm)
        #             rew = []
        #             for f in features:
        #                 rew += self.rewards_per_feature[arm][i][f[0]][f[1]]
        #             rewards.append(rew)
        #             if len(rew) > 0:
        #                 pulled_arms.append(arm)
        #             else:
        #                 pulled_arms.append(-1)
        #         self.mab.update(pulled_arms, rewards)

        if len(self.rewards_per_feature) > 0:
            for i in range(self.n_learners):
                x = []
                y = []

                for user in self.rewards_per_feature:
                    if user[3] in self.features and user[1] == i:
                        bud = user[2] * self.get_users_in_context(self.features) / self.get_users_in_context([user[3]])
                        if bud <= 100:
                            x.append(bud)
                            y.append(user[0])

                x = np.array(x)
                y = np.array(y)

                x, y = self.unison_shuffled_copies(x, y)

                x = x.tolist()
                y = y.tolist()

                if not (len(x) == 0 or len(y) == 0):
                    # x = np.atleast_2d(x).T
                    self.mab.update_single(i, x, y)

                        






            # for arm in range(self.n_arms):
            #     rewards = []
            #     pulled_arms = []
            #     for i in range(self.n_learners):
            #         #pulled_arms.append(arm)
            #         rew = []
            #         self.rewards_per_feature



            #         if len(rew) > 0:
            #             pulled_arms.append(arm)
            #         else:
            #             pulled_arms.append(-1)
            #     self.mab.update(pulled_arms, rewards)
                    

    
    def update(self, arms, rewards, user_features):
        context_reward = []
        for i in range(self.n_learners):
            context_reward_per_arm = []
            for r, f in zip(rewards[i], user_features):
                if f in self.features:
                    context_reward_per_arm.append(r)
            context_reward.append(context_reward_per_arm)
        # print("context arm:", arms)
        self.mab.update(arms, context_reward)


    def get_expected_rewards(self):
        return self.mab.get_expected_rewards()

    def get_quantities(self):
        return self.quantity_estimator.get_quantities(self.features)

    def get_users_in_context(self, feature_list):
        tot = 0
        for feature in self.translate_features(feature_list):
            tot += self.average_users_per_feature[feature]
        return tot

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