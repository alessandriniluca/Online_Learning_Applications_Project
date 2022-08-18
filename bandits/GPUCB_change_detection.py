from bandits.gpucb1 import GPUCB1_Learner
from change_detection.change_detection import ChangeDetection
import numpy as np


class GPUCBChangeDetection(GPUCB1_Learner):

  def __init__(self, n_arms, arms, M, eps, h, name=""):
    super().__init__(n_arms, arms, name)
    self.change_detection = ChangeDetection(n_arms, M, eps, h)
    self.valid_rewards = [[] for arm in range(n_arms)]

  def update_observations(self, arm_idx, reward):
    self.this_round_rewards.append(reward)


  def update_model(self):
    """
    This method update internal model retraining the GP and predicting again the values for every arm
    """

    if self.change_detection.update(self.this_round_arm_idx, np.average(self.this_round_rewards)):
        self.valid_rewards[self.this_round_arm_idx] = []
    else:
        self.valid_rewards[self.this_round_arm_idx] += self.this_round_rewards

    arms_list = []
    rewards = []
    for arm_idx, single_arm_rewards in enumerate(self.valid_rewards):
      for reward in single_arm_rewards:
        arms_list.append(self.arms[arm_idx])
        rewards.append(reward)

    x = np.atleast_2d(arms_list).T 
    y = np.array(rewards)

    # Retrain the GP
    self.gp.fit(x, y)

    # Retrieve predictions from GP
    self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)

    # sigma lower bound
    self.sigmas = np.maximum(self.sigmas, 1e-2)

  def update(self, pulled_arm, reward):
    """
      This method update the GP state and internal model
    """
    self.t += 1

    self.this_round_rewards = []
    self.this_round_arm_idx = pulled_arm

    for r in reward:
      self.update_observations(pulled_arm, r)
    self.update_model()
  