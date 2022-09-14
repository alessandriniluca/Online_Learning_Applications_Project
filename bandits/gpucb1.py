from math import sqrt, log
from random import shuffle

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from bandits.learner import Learner


class GPUCB1_Learner(Learner):
    """
    GPUCB implementation
    """

    def __init__(self, n_arms, arms, name=""):
        super().__init__(n_arms, name)
        self.arms = arms
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms) * 1.0
        self.pulled_arms = []
        alpha = .5
        kernel = C(1, constant_value_bounds="fixed") * RBF(50, length_scale_bounds="fixed")

        # kernel = 1 * RBF(length_scale=2.0, length_scale_bounds=(1e-2, 1e2))
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha ** 2,
            normalize_y=True,
            n_restarts_optimizer=9
        )

    def update_observations(self, arm_idx, reward):
        """
        Update internal state given last action and its outcome
        """
        #
        super().update_observations(arm_idx, reward)
        self.pulled_arms.append(self.arms[arm_idx])

    def update_model(self):
        """
        This method update internal model retraining the GP and predicting again the values for every arm
        """
        # Prepare X, y for GP
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards

        # Retrain the GP
        self.gp.fit(x, y)

        # Retrieve predictions from GP
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)

        # sigma lower bound
        #self.sigmas = np.maximum(self.sigmas, 1e-2)

    def update(self, pulled_arm, reward):
        """
        This method update the GPTS state and internal model
        """
        self.t += 1
        for r in reward:
            self.update_observations(pulled_arm, r)
        self.update_model()

    def get_expected_rewards(self):
        """
        Return expected rewards that will be provided to an optimizer to complete the combinatorial Bandit
        """
        beta = np.sqrt(2*np.log2(self.n_arms*(self.t+1)*(self.t+1)*3.14*3.14/(6*0.05)))
        # print("---- beta", beta, self.sigmas)
        upper_bounds = self.means + beta * self.sigmas

        upper_bounds = np.clip(upper_bounds, 0, 1)

        return upper_bounds
