import numpy as np
from matplotlib import pyplot as plt

from bandits.gpts import GPTS_Learner
from bandits.gpucb1 import GPUCB1_Learner
from bandits.multi_learner import MultiLearner
from change_detection.alpha_function_generator import AlphaFunctionGenerator
from common.utils import load_static_env_configuration, load_static_sim_configuration, get_test_alphas_functions, \
    LearnerType, save_data
from environment.environment import Environment
from optimizer.estimator import Estimator
from optimizer.full_optimizer import FullOptimizer
from optimizer.optimizer import Optimizer

np.set_printoptions(formatter={'float': lambda x: "{0:0.10f}".format(x)})


env_configuration = load_static_env_configuration("configurations/environment/static_conf_1.json")
sim_configuration = load_static_sim_configuration("configurations/simulation/sim_conf_1.json")
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


TIME_HORIZON = 60
N_EXPERIMENTS = 3
N_CAMPAIGNS = 5
CHANGE_FUNCTION_PERIOD = 30



n_arms = int(sim_configuration["total_budget"] / sim_configuration["resolution"]) + 1
budgets = np.linspace(0, sim_configuration["total_budget"], n_arms)

mean_profit = []
mean_regret = []



for e in range(0, N_EXPERIMENTS):
    best_expected_profits = []
    best_allocation_res = []
    # Initialize a bandits to estimate alpha functions
    # TODO forse meglio gestire due simulazioni differenti, una per TS e una per UCB
    #      meglio fissare un seed per i generatori random così da poter riprodurre e confrontare
    #      gli esperimenti
    #gpucb_learners = MultiLearner(n_arms, budgets, LearnerType.UCB1, n_learners=N_CAMPAIGNS)
    gpts_learners = MultiLearner(n_arms, budgets, LearnerType.UCB_SLIDING_WINDOW, n_learners=N_CAMPAIGNS)

    env = Environment(
        configuration=env_configuration,
        alphas_functions=alphas_functions
    )
    profits = []
    season = 0

    function_generator = AlphaFunctionGenerator()

    for t in range(TIME_HORIZON):

        if t == 0 or t == 30:

            env.alphas_functions = function_generator.get_functions(season)
            season += 1

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
                alphas_functions=env.alphas_functions,
                one_campaign_per_product=True
            )

            # Optimize 5 campaigns with all data known to compute the baseline
            print("=== OPTIMIZER STARTED ===")
            optimizer.one_campaign_per_product = True
            optimizer.run_optimization()
            best_allocation, best_expected_profit = optimizer.find_best_allocation()
            print("=== THIS IS OPTIMAL ALLOCATION ===")
            print(best_allocation)
            best_allocation_res.append(best_allocation)

        best_expected_profits.append(best_expected_profit)
        
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
        current_allocation, expected_profit = optimizer.find_best_allocation()
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
                if reward_index == user.starting_product:
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
        # print("INDICE::::", arm_indexes)
        # print("Rewards:::", rewards)
        gpts_learners.update(arm_indexes, rewards)
        profits.append(round_profit)

    # TODO end of simulation, compare result and analyze regret vs clairvoyant
    #      Note that best expected profit is just the average so it may be overtaken by sub optimal
    #      allocation due to variance [We could cope with that considering also variance and take upperbound]
    regrets = []

    for i in range(len(profits)):
        regrets.append(best_expected_profits[i] - profits[i])

    mean_profit.append(profits)
    mean_regret.append(regrets)

# print("REG:", mean_regret)
# print("PROF:", mean_profit)
print(best_allocation_res)

save_data("task6_sliding_window",
    [
    "experiments: "+str(N_EXPERIMENTS),
    "rounds: "+str(TIME_HORIZON),
    "regret", list(np.mean(mean_regret, axis=0)), 
    "profit", list(np.mean(mean_profit, axis=0)),
    "std_dev", list(np.std(mean_profit, axis=0))]
    )

plt.figure(0)
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot(np.mean(mean_regret, axis=0), 'r')

plt.legend(["REGRET"])
plt.show()

plt.figure(1)
plt.ylabel("Profit")
plt.xlabel("t")
plt.plot(np.mean(mean_profit, axis=0), 'g')
# plt.axhline(y=best_expected_profit, color='b', linestyle='-')
plt.plot(best_expected_profits, 'b')


plt.legend(["PROFIT", "OPTIMAL AVG"])
plt.show()

if __name__ == '__main__':
    print("simulation done")
