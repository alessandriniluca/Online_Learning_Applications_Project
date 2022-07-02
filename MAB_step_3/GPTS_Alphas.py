from MAB_step_3.GPTS_Learner import GPTS_Learner
import numpy as np


class GPTS_Alphas:
    def __init__(self, n_arms, arms, products=5):
        """
    n_arms: number of budgets we want to try
    arms: values of the budget to try
    """
        self.n_arms = max(n_arms)
        self.products_number = products
        self.regressors = []
        for i in range(5):
            self.regressors.append(GPTS_Learner(n_arms[i], arms[i], "learner_" + str(i)))

    def pull_arms(self):
        alphas_prime = np.zeros((self.n_arms, self.products_number, 1))
        for i in range(len(self.regressors)):
            alphas_prime[:, i, 0] = self.regressors[i].pull_arm()
        # print("My alphas:",alphas_prime)

        return alphas_prime

    def update(self, pulled_arms, rewards):
        i = 0
        for arm in pulled_arms:
            self.regressors[i].update_step()
            at_least_one = False
            for reward in rewards:
                at_least_one = True
                self.regressors[i].update_observations(arm, 1 if reward == i else 0)
            if at_least_one:
                self.regressors[i].update_model()
            i += 1
