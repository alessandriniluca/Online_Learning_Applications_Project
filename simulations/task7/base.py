import numpy as np
from matplotlib import pyplot as plt

from context_generation.context_generator import ContextGenerator

from common.utils import load_static_env_configuration, load_static_sim_configuration, get_test_alphas_functions, \
    LearnerType, save_data, translate_feature_group
from environment.environment_context import Environment
from optimizer.estimator import Estimator
from optimizer.full_optimizer_context import FullOptimizer
from optimizer.optimizer_context import Optimizer

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
    features_division=[[0], [1], [2,3]],
    one_campaign_per_product=False
)

# Optimize 5 campaigns with all data known to compute the baseline
print("=== OPTIMIZER STARTED task7===")
optimizer.one_campaign_per_product = False
optimizer.run_optimization()
best_allocation, best_expected_profit = optimizer.find_best_allocation()
print("=== THIS IS OPTIMAL ALLOCATION ===")
print(best_allocation, best_expected_profit)

# Start simulation estimating alpha functions

TIME_HORIZON = 52
N_EXPERIMENTS = 20
N_CAMPAIGNS = 5

n_arms = int(sim_configuration["total_budget"] / sim_configuration["resolution"]) + 1
budgets = np.linspace(0, sim_configuration["total_budget"], n_arms)

mean_profit = []
mean_regret = []


for e in range(0, N_EXPERIMENTS):
    # Initialize a bandits to estimate alpha functions
    # TODO forse meglio gestire due simulazioni differenti, una per TS e una per UCB
    #      meglio fissare un seed per i generatori random così da poter riprodurre e confrontare
    #      gli esperimenti
    #gpucb_learners = MultiLearner(n_arms, budgets, LearnerType.UCB1, n_learners=N_CAMPAIGNS)
    # print("EEEEEEE", budgets)
    context_generator = ContextGenerator(n_arms, budgets, LearnerType.UCB1, env_configuration.average_users_number)
    context_generator.start() # crea un unico context con un unico bandit
    env = Environment(
        configuration=env_configuration,
        alphas_functions=alphas_functions
    )
    profits = []

    print("-----*******-------- EXPERIMENT NUMBER: ", e)

    for t in range(TIME_HORIZON):

        if t % 14 == 0 and t > 1:
            context_generator.split()
        
        # Ask for estimations (get alpha primes)
        contexts, ts_alpha_prime = context_generator.get_expected_rewards()

        # Run optimization
        optimizer = Optimizer(
            users_number=env.configuration.average_users_number,
            min_budget=sim_configuration["min_budget"],
            max_budget=sim_configuration["max_budget"],
            total_budget=sim_configuration["total_budget"],
            resolution=sim_configuration["resolution"],
            products=env.products,
            mean_quantities=context_generator.get_quantities(),
            buy_probs=buy_probs,
            alphas=ts_alpha_prime,
            features_division=translate_feature_group(contexts),
            one_campaign_per_product=False
        )

        optimizer.run_optimization()
        current_allocation, expected_profit = optimizer.find_best_allocation()
        print(current_allocation)

        # Compute Rewards from the environment
        round_users, feature0_escaped, feature1_escaped, feature2_escaped, feature3_escaped, round_profit = env.round(current_allocation, translate_feature_group(contexts))

        # Update the learners - 0/1 implementation
        # Note: as discussed we try to provide to the learner 1 if a user started from corresponding product
        #       otherwise the learner get a 0.
        # TODO This approach needs to be tested and compared with direct approach (estimate alpha as buyers/tot directly)
        #      Note that this approach is not so efficient since require more of computation and memory
        rewards = [[], [], [], [], []]
        user_features = []
        for user in round_users:
            # compute the rewards
            user_features.append(user.features)
            for reward_index, reward in enumerate(rewards):
                # if the index match reward is 1
                if reward_index == user.starting_product:
                    reward.append(1)
                # otherwise zero
                else:
                    reward.append(0)

        # fill the rewards to compensate lost users
        for i in range(feature0_escaped):
            user_features.append((0,0))
            for j in range(len(rewards)):
                rewards[j].append(0)

        for i in range(feature1_escaped):
            user_features.append((0,1))
            for j in range(len(rewards)):
                rewards[j].append(0)

        for i in range(feature2_escaped):
            user_features.append((1,0))
            for j in range(len(rewards)):
                rewards[j].append(0)

        for i in range(feature3_escaped):
            user_features.append((1,1))
            for j in range(len(rewards)):
                rewards[j].append(0)
        
        # compute index of arm played
        # TODO this could be prevented making update method work also with value (not only index)
        arm_indexes = []
        for allocation in current_allocation:
            arm_indexes.append(np.where(budgets == allocation)[0][0])
        # update the learners
        # print("INDICE::::", arm_indexes)
        # print("Rewards:::", rewards)
        context_generator.update_quantities(round_users)
        context_generator.update(arm_indexes, rewards, user_features)
        profits.append(round_profit)

    # TODO end of simulation, compare result and analyze regret vs clairvoyant
    #      Note that best expected profit is just the average so it may be overtaken by sub optimal
    #      allocation due to variance [We could cope with that considering also variance and take upperbound]
    regrets = []
    for profit in profits:
        regrets.append(best_expected_profit - profit)

    mean_profit.append(profits)
    mean_regret.append(regrets)

# print("REG:", mean_regret)
# print("PROF:", mean_profit)

save_data("task7_classes",
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
