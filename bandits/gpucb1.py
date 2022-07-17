from math import sqrt, log

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from bandits.learner import Learner


class GPUCB1_Learner(Learner):
    """
    GPUCB implementation
    """
    def __init__(self, n_arms, arms):
        super().__init__(n_arms)
        self.arms = arms
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms) * 8
        self.pulled_arms = []
        alpha = 1.0
        kernel = C(1, constant_value_bounds="fixed") * RBF(2, length_scale_bounds="fixed")
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
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def update(self, pulled_arm, reward):
        """
        This method update the GPTS state and internal model
        """
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

    def get_expected_rewards(self):
        """
        Return expected rewards that will be provided to an optimizer to complete the combinatorial Bandit
        """
        upper_bounds = np.zeros(len(self.arms))
        for arm_index, arm in enumerate(self.arms):
            upper_bound = self.means[arm_index]
            upper_bound += sqrt((2*log(self.t))/self.pulled_arms.count(arm))
            upper_bounds[arm_index] = upper_bound
        return upper_bounds
