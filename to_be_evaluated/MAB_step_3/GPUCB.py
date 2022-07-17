from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np
from bandits.learner import Learner


class GPUCB(Learner):
    def __init__(self, n_arms, arms, name):
        super().__init__(n_arms)
        
        self.name = name
        self.arms = arms
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms) * 10
        self.pulled_arms = []

        kernel = C(1, constant_value_bounds="fixed") * RBF(4, length_scale_bounds="fixed")

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.5,
            normalize_y=True,
            n_restarts_optimizer=9  # Parametri per euristica
        )

    def update_observations(self, arm_idx, reward):
        """
      Update the internal state after played an arm given a reward

      Args:
          arm_idx: index of played arm
          reward: reward observed after the round
    """
        super().update_observations(arm_idx, reward)

        self.pulled_arms.append(self.arms[arm_idx])

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards

        print("Learner mean:", self.name, ":", np.mean(self.collected_rewards))

        self.gp.fit(x, y)

        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)

        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def update(self, pulled_arm, reward):
        # self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

    def update_step(self):
        self.t += 1

    def pull_arm(self):
        pass
