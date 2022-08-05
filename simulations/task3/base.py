import numpy as np

from bandits.gpts import GPTS_Learner
from bandits.gpucb1 import GPUCB1_Learner
from bandits.multi_learner import MultiLearner
from common.utils import load_static_env_configuration, load_static_sim_configuration, get_test_alphas_functions, \
    LearnerType
from environment.environment import Environment
from optimizer.estimator import Estimator
from optimizer.full_optimizer import FullOptimizer
from optimizer.optimizer import Optimizer

np.set_printoptions(formatter={'float': lambda x: "{0:0.10f}".format(x)})


env_configuration = load_static_env_configuration("../../configurations/environment/static_conf_1.json")
sim_configuration = load_static_sim_configuration("../../configurations/simulation/sim_conf_1.json")
alphas_functions = get_test_alphas_functions()

env = Environment(
    configuration=env_configuration,
    alphas_functions=alphas_functions
)

estimator = Estimator(env.configuration.graph_clicks,
                      env.products,
                      env.configuration.lambda_prob,
                      env.configuration.reservation_price_means,
                      env.configuration.reservation_price_std_dev
                      )

buy_probs = estimator.get_buy_probs()

optimizer = FullOptimizer(
    users_number=env.configuration.average_users_number,
    min_budget=sim_configuration["min_budget"],
    max_budget=sim_configuration["max_budget"],
    total_budget=sim_configuration["total_budget"],
    resolution=sim_configuration["resolution"],
    products=env.products,
    mean_quantities=env.configuration.quantity_means,
    buy_probs=buy_probs,
    basic_alphas=env.configuration.basic_alphas,
    alphas_functions=alphas_functions,
    one_campaign_per_product=True
)

# Optimize 5 campaigns with all data known to compute the baseline
optimizer.one_campaign_per_product = True
optimizer.run_optimization()
best_allocation = optimizer.find_best_allocation()
print(best_allocation)

# Start simulation estimating alpha functions

TIME_HORIZON = 100
N_EXPERIMENTS = 1
N_CAMPAIGNS = 5

n_arms = int(sim_configuration["total_budget"] / sim_configuration["resolution"]) + 1
budgets = np.linspace(0, sim_configuration["total_budget"], n_arms)

for e in range(0, N_EXPERIMENTS):
    # Initialize a bandits to estimate alpha functions
    # TODO forse meglio gestire due simulazioni differenti, una per TS e una per UCB
    #      meglio fissare un seed per i generatori random così da poter riprodurre e confrontare
    #      gli esperimenti
    # gpucb_learners = MultiLearner(n_arms, budgets, LearnerType.UCB1, n_learners=n_campaigns)
    gpts_learners = MultiLearner(n_arms, budgets, LearnerType.TS, n_learners=N_CAMPAIGNS)

    for t in range(TIME_HORIZON):
        # Ask for estimations (get alpha primes)
        ts_alpha_prime = gpts_learners.get_expected_rewards()

        # Run optimization
        optimizer = Optimizer(
            users_number=env.configuration.average_users_number,
            min_budget=sim_configuration["min_budget"],
            max_budget=sim_configuration["max_budget"],
            total_budget=sim_configuration["total_budget"],
            resolution=sim_configuration["resolution"],
            products=env.products,
            mean_quantities=env.configuration.quantity_means,
            buy_probs=buy_probs,
            alphas=ts_alpha_prime,
            one_campaign_per_product=True
        )

        optimizer.run_optimization()
        current_allocation = optimizer.find_best_allocation()
        print(current_allocation)

        # Compute Rewards from the environment
        round_users, total_users, round_profit = env.round(current_allocation)

        # Update the learners - 0/1 implementation
        # Note: as discussed we try to provide to the learner 1 if a user started from corresponding product
        #       otherwise the learner get a 0.
        # TODO This approach needs to be tested and compared with direct approach (estimate alpha as buyers/tot directly)
        #      Note that this approach is not so efficient since require more of computation and memory
        rewards = [[], [], [], [], []]
        for user in round_users:
            # compute the rewards
            for reward_index, reward in enumerate(rewards):
                # if the index match reward is 1
                if reward_index == user.seen_product[0].number:
                    reward.append(1)
                # otherwise zero
                else:
                    reward.append(0)

        # Array with one zero for each lost customer
        lost_users_reward = np.zeros(total_users - len(round_users))
        # fill the rewards to compensate lost users
        for i in range(len(rewards)):
            rewards[i] = np.concatenate((rewards[i], lost_users_reward))

        # compute index of arm played
        # TODO this could be prevented making update method work also with value (not only index)
        arm_indexes = []
        for allocation in current_allocation:
            arm_indexes.append(np.where(budgets == allocation)[0][0])

        # update the learners
        gpts_learners.update(arm_indexes, rewards)

    # TODO end of simulation, compare result and analyze regret vs clairvoyant

if __name__ == '__main__':
    print("simulation done")
