from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np
from bandits.learner import Learner
from matplotlib import pyplot as plt


class GPTS_Learner(Learner):
    def __init__(self, n_arms, arms, name):
        super().__init__(n_arms)
        self.name = name
        self.arms = arms
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms) * 10
        self.pulled_arms = []
        # kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        kernel = C(1, constant_value_bounds="fixed") * RBF(4, length_scale_bounds="fixed")
        # kernel = kernel = ConstantKernel(0.5) + Matern(length_scale=1, nu=3/2) + WhiteKernel(noise_level=0.02)

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.5,
            normalize_y=True,
            n_restarts_optimizer=9  # Parametri per euristica
        )

    def update_observations(self, arm_idx, reward):
        super().update_observations(arm_idx, reward)

        self.pulled_arms.append(self.arms[arm_idx])

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards

        print("Learner mean:", self.name, ":", np.mean(self.collected_rewards))

        self.gp.fit(x, y)
        if self.t == -25 or self.t == -50:
            x_pred = np.atleast_2d(self.arms).T
            y_pred, sigma = self.gp.predict(x_pred, return_std=True)

        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)

        self.sigmas = np.maximum(self.sigmas, 1e-2)

        if self.t == -25 or self.t == -50:
            # print("Learner:", self.name, ":", self.means, self.sigmas)

            # print("means:", self.means, "sigmas:", self.sigmas)
            # print("X:", x)
            # print("Y:", y)
            plt.figure(self.t)
            plt.title(f'Iteration {self.t} {self.name}')
            plt.plot(x.ravel(), y, 'ro', label=r'Observed Clicks')
            plt.plot(x_pred, y_pred, 'b-', label=r'Predicted clicks')
            plt.fill(
                np.concatenate([x_pred, x_pred[::-1]]),
                np.concatenate([y_pred - 1.96 * sigma, (y_pred + 1.96 * sigma)[::-1]]),
                alpha=.5, fc='b', ec='None', label='95% conf interval')
            plt.xlabel('$x$')
            plt.ylabel('$n(x)$')
            plt.legend(loc='lower right')
            plt.show()

    def update(self, pulled_arm, reward):
        # self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

    def update_step(self):
        self.t += 1

    def pull_arm(self):
        sampled_values = np.random.normal(self.means, self.sigmas)
        for i in range(len(sampled_values)):
            if sampled_values[i] <= 0:
                sampled_values[i] = 0.0000000001
            elif sampled_values[i] >= 1:
                sampled_values[i] = 0.9999999999
        return sampled_values
