import numpy as np

from bandits.multi_learner import MultiLearner
from probability_calculator.quantities_estimator import QuantitiesEstimator

class Context:

    def __init__(self, quantity_estimator, rewards_per_feature, n_arms, budgets, learner_type, n_learners, features):
        self.rewards_per_feature = rewards_per_feature
        self.n_arms = n_arms
        self.budgets = budgets
        self.learner_type = learner_type
        self.n_learners = n_learners
        self.features = features
        self.quantity_estimator = quantity_estimator
        self.mab = MultiLearner(self.n_arms, self.budgets, self.learner_type, self.n_learners)


        if len(self.rewards_per_feature) > 0:
            for arm in range(self.n_arms):
                rewards = []
                pulled_arms = []
                for i in range(self.n_learners):
                    #pulled_arms.append(arm)
                    rew = []
                    for f in features:
                        rew += self.rewards_per_feature[arm][i][f[0]][f[1]]
                    rewards.append(rew)
                    if len(rew) > 0:
                        pulled_arms.append(arm)
                    else:
                        pulled_arms.append(-1)
                self.mab.update(pulled_arms, rewards)
                    

    
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