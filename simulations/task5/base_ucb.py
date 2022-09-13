import numpy as np

from matplotlib import pyplot as plt
from bandits.multi_learner import MultiLearner
from common.utils import LearnerType, load_static_env_configuration, load_static_sim_configuration, get_test_alphas_functions, save_data
from environment.configuration import Configuration
from environment.environment import Environment
from environment.environment_complete_history import EnvironmentCompleteHistory
from optimizer.estimator import Estimator
from optimizer.full_optimizer import FullOptimizer
from optimizer.optimizer import Optimizer
from probability_calculator.uncertain_graph_weights_estimator import GraphWeightsEstimator
from common.utils import get_logger, get_products

np.set_printoptions(formatter={'float': lambda x: "{0:0.10f}".format(x)})

logger = get_logger(__name__)


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
print("=== OPTIMIZER STARTED task5===")
optimizer.one_campaign_per_product = True
optimizer.run_optimization()
best_allocation, best_expected_profit = optimizer.find_best_allocation()
print("=== THIS IS OPTIMAL ALLOCATION ===")
print(best_allocation)


######################
TIME_HORIZON = 35
N_EXPERIMENTS = 10
N_CAMPAIGNS = 5
test_experiments = []

env = EnvironmentCompleteHistory(
    configuration=env_configuration,
    alphas_functions=alphas_functions
)

graph_estimator = GraphWeightsEstimator(env.configuration.lambda_prob, len(env.products), env.products)

estimator = Estimator(graph_estimator.get_estimated_graph(),
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
    alphas_functions=alphas_functions
)

mean_profit = []
mean_regret = []

n_arms = int(sim_configuration["total_budget"] / sim_configuration["resolution"]) + 1
budgets = np.linspace(0, sim_configuration["total_budget"], n_arms)

for experiment in range(N_EXPERIMENTS):
    profit = []
    regret = []

    gpucb_learners = MultiLearner(n_arms, budgets, LearnerType.UCB1, n_learners=N_CAMPAIGNS)
    estimator.update_graph_clicks(graph_estimator.get_estimated_graph())
    print("EEE", estimator.get_buy_probs())
    print("-----*******-------- EXPERIMENT NUMBER: ", experiment)

    for round in range(TIME_HORIZON):

        #set probabilities
        buy_probs = estimator.get_buy_probs()
        optimizer.set_buy_probabilities(buy_probs)
        ucb_alpha_prime = gpucb_learners.get_expected_rewards()



        optimizer = Optimizer(
            users_number=env.configuration.average_users_number,
            min_budget=sim_configuration["min_budget"],
            max_budget=sim_configuration["max_budget"],
            total_budget=sim_configuration["total_budget"],
            resolution=sim_configuration["resolution"],
            products=env.products,
            mean_quantities=env.configuration.quantity_means,
            buy_probs=buy_probs,
            alphas=ucb_alpha_prime,
            one_campaign_per_product=True
        )

        #Optimize with probabilities calculated
        optimizer.one_campaign_per_product = True
        optimizer.run_optimization()
        current_allocation, expected_earning = optimizer.find_best_allocation()
        print(current_allocation)

        # end of optimization
        # perform the round and get the history of the users
        users_history, round_profit, round_users, total_users = env.round(current_allocation)
        # Update graph probabilities according to users history
        graph_estimator.update_graph_probabilities(users_history)
        estimator.update_graph_clicks(graph_estimator.get_estimated_graph())
        print(current_allocation)

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
        print(total_users)
        print(len(round_users))
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
        gpucb_learners.update(arm_indexes, rewards)

        profit.append(round_profit)
        regret.append(best_expected_profit - round_profit)
    mean_profit.append(profit)
    mean_regret.append(regret)
    
    # For each experiment, we have to zero everything to repeat the estimation
    test_experiments.append(graph_estimator.get_estimated_graph().copy())
    graph_estimator.reset()
    logger.info("Best allocation for experiment: " + str(best_allocation))
    logger.info("Reward: ")

print("\n\n\n")
for i, e in enumerate(test_experiments):
    logger.info("\n Experiment numeber " + str(i) + ". Results:\n" + str(e))



# print("REG:", mean_regret)
# print("PROF:", mean_profit)
save_data("task5_graph",
    {
        "experiments": N_EXPERIMENTS,
        "rounds": TIME_HORIZON,
        "regrets": mean_regret,
        "profits": mean_profit,
        "best_expected_profit": best_expected_profit,
        "regret_means": np.mean(mean_regret, axis=0).tolist(), 
        "profit_means": np.mean(mean_profit, axis=0).tolist(),
        "profit_means_std_dev": np.std(mean_profit, axis=0).tolist()
    }
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
plt.axhline(y=best_expected_profit, color='b', linestyle='-')

plt.legend(["PROFIT", "OPTIMAL AVG"])
plt.show()

if __name__ == '__main__':
    print("simulation done")
