import numpy as np
from bandits.gpucb1 import GPUCB1_Learner

class GPUCBSlidingWindow(GPUCB1_Learner):

  def __init__(self, n_arms, arms, window_size, name=""):
    super().__init__(n_arms, arms, name)
    self.valid_rewards = []
    self.window_size = window_size
    self.time_elapsed = 0
    self.n_arms = n_arms



  def update_observations(self, arm_idx, reward):
    self.valid_rewards.append( (arm_idx, reward) )
    # print(self.valid_rewards)


  def update_model(self):
    """
    This method update internal model retraining the GP and predicting again the values for every arm
    """

    arms_list = []
    rewards = []

    for day in self.valid_rewards:
      arm_idx = day[0]
      rewards_list = day[1]

      for reward in rewards_list:
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
    self.t += 1
    self.update_observations(pulled_arm, reward)
    self.time_elapsed += 1

    if self.time_elapsed > self.window_size:
      self.valid_rewards = self.valid_rewards[1:]

    self.update_model()

