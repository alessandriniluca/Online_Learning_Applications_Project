from bandits.GPUCB_sliding_window import GPUCBSlidingWindow
from bandits.GPUCB_change_detection import GPUCBChangeDetection
from bandits.gpucb1 import GPUCB1_Learner
from bandits.learner import Learner
from common.utils import LearnerType
from bandits.gpts import GPTS_Learner
import numpy as np


class MultiLearner:
    def __init__(self, n_arms, arms, learner_type, n_learners=5):
        """
    n_arms: number of budgets we want to try
    arms: values of the budget to try
    """
        self.n_arms = n_arms
        self.n_learners = n_learners
        self.learner_type = learner_type
        self.learners = []
        for i in range(n_learners):
            if learner_type == LearnerType.TS:
                self.learners.append(GPTS_Learner(n_arms, arms, "learner_" + str(i)))
            elif learner_type == LearnerType.UCB1:
                self.learners.append(GPUCB1_Learner(n_arms, arms, "learner_" + str(i)))
            elif learner_type == LearnerType.UCB_CHANGE_DETECTION:
                self.learners.append(GPUCBChangeDetection(n_arms, arms, 3, 0.03, .075, "learner_" + str(i)))
            elif learner_type == LearnerType.UCB_SLIDING_WINDOW:
                self.learners.append(GPUCBSlidingWindow(n_arms, arms, 9, "learner_" + str(i)))


    def get_expected_rewards(self):
        alphas_prime = np.zeros((self.n_arms, self.n_learners, 1))
        for i in range(len(self.learners)):
            alphas_prime[:, i, 0] = self.learners[i].get_expected_rewards()
        return alphas_prime

    def update(self, pulled_arms, rewards):
        """
        Args:
            pulled_arms (list): list of the arms pulled (5 elements in aggregate case, 15 otherwise)
            rewards (list of lists): the i-th sub-list will contain the rewards for bandit i
        """
        # print("_-----------_")
        # print(rewards)
        # print(len(self.learners))
        # print("PULLED ARM:", pulled_arms)
        for i, arm in enumerate(pulled_arms):
            if arm >= 0:
                self.learners[i].update(arm, rewards[i])
