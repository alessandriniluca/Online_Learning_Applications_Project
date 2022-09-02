import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C

from bandits.learner import Learner
from matplotlib import pyplot as plt



class GPTS_Learner(Learner):
    """
    GPTS implementation
    """
    def __init__(self, n_arms, arms, name=""):
        super().__init__(n_arms, name)
        self.arms = arms
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms) * 8
        self.pulled_arms = []
        alpha = .5
        # kernel = C(5, constant_value_bounds="fixed") * RBF(20, length_scale_bounds="fixed")
        kernel = Matern(length_scale=1.0, length_scale_bounds="fixed", nu=1.5)
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

        # x_pred = np.atleast_2d(self.arms).T
        # y_pred, sigma = self.gp.predict(x_pred, return_std=True)
        # plt.figure(self.t)
        # plt.ylim(-0.5, 1.5)
        # plt.title(f'Iteration {self.t} {self.name}')
        # plt.plot(x.ravel(), y, 'ro', label=r'Observed Clicks')
        # plt.plot(x_pred, y_pred, 'b-', label=r'Predicted clicks')
        # plt.fill(
        #     np.concatenate([x_pred, x_pred[::-1]]), 
        #     np.concatenate([y_pred - 1.96 * sigma, (y_pred + 1.96 * sigma)[::-1]]),
        #     alpha=.5, fc='b', ec='None', label='95% conf interval')
        # plt.xlabel('$x$')
        # plt.ylabel('$n(x)$')
        # plt.legend(loc='lower right')
        # plt.show()
        

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
        sampled_values = np.random.normal(self.means, self.sigmas)
        print("Learner ", self.name, "sampled values:", sampled_values)
        for i in range(len(sampled_values)):
            sampled_values[i] = min(max(sampled_values[i], 0.0), 1.0)
        return sampled_values
