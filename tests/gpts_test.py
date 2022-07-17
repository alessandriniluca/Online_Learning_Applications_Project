import numpy as np
from matplotlib import pyplot as plt

from bandits.gpts import GPTS_Learner
from bandits.gpucb1 import GPUCB1_Learner

np.set_printoptions(formatter={'float': lambda x: "{0:0.10f}".format(x)})


def fun(x):
    return (200.0 * (1.0 - np.exp(-0.02 * x))).astype(int)


class SimpleTestEnv:
    def __init__(self, budget, sigma):
        self.bids = budget
        self.means = fun(budget)
        self.sigmas = np.ones(len(budget)) * sigma

    def round(self, pulled_arm):
        return np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm])


n_arms = 20
min_budget = 0
max_budget = 180
budgets = np.linspace(min_budget, max_budget, n_arms)
sigma = 2

T = 200
n_experiments = 10

gpts_rewards_per_experiment = []
gpucb_rewards_per_experiment = []

# Run the tests
for e in range(0, n_experiments):
    # Init the env and the GPTS
    env = SimpleTestEnv(budgets, sigma)
    gpts_learner = GPTS_Learner(n_arms=n_arms, arms=budgets)
    gpucb_learner = GPUCB1_Learner(n_arms=n_arms, arms=budgets)

    for t in range(T):
        # Get expected reward for each arm
        expected_rewards = gpts_learner.get_expected_rewards()
        arm_to_play = np.argmax(expected_rewards)
        # Play the selected arm
        reward = env.round(arm_to_play)
        reward = reward - budgets[arm_to_play]
        # Update the learner
        gpts_learner.update(arm_to_play, reward)

        # Get expected reward for each arm
        arm_to_play = 0
        if gpucb_learner.t < len(gpts_learner.arms):
            arm_to_play = gpucb_learner.t
        else:
            expected_rewards = gpucb_learner.get_expected_rewards()
            arm_to_play = np.argmax(expected_rewards)
        # Play the selected arm
        reward = env.round(arm_to_play)
        reward = reward - budgets[arm_to_play]
        # Update the learner
        gpucb_learner.update(arm_to_play, reward)

    gpts_rewards_per_experiment.append(gpts_learner.collected_rewards)
    gpucb_rewards_per_experiment.append(gpucb_learner.collected_rewards)

# Get optimal
opt = np.max(env.means - budgets)
optimals = np.ones(T) * opt

plt.figure(0)
plt.ylabel("Profit")
plt.xlabel("t")
plt.plot(np.cumsum(optimals), 'g')
plt.plot(np.cumsum(np.mean(gpts_rewards_per_experiment, axis=0)), 'r')
plt.plot(np.cumsum(np.mean(gpucb_rewards_per_experiment, axis=0)), 'b')

plt.legend(["OPTIMAL", "GPTS", "GPUCB1"])
plt.show()

plt.figure(1)
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot(np.cumsum(np.mean(opt - gpts_rewards_per_experiment, axis=0)), 'r')
plt.plot(np.cumsum(np.mean(opt - gpucb_rewards_per_experiment, axis=0)), 'b')

plt.legend(["GPTS", "GPUCB1"])
plt.show()

print(env.means - budgets)
