import numpy as np

from matplotlib import pyplot as plt
from common.utils import load_static_env_configuration, load_static_sim_configuration, get_test_alphas_functions, save_data
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

TIME_HORIZON = 150
N_EXPERIMENTS = 5
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

for experiment in range(N_EXPERIMENTS):
    profit = []
    regret = []
    estimator.update_graph_clicks(graph_estimator.get_estimated_graph())

    for round in range(TIME_HORIZON):

        #set probabilities
        buy_probs = estimator.get_buy_probs()
        optimizer.set_buy_probabilities(buy_probs)

        #Optimize with probabilities calculated
        optimizer.one_campaign_per_product = True
        optimizer.run_optimization()
        best_allocation, expected_earning = optimizer.find_best_allocation()
        # end of optimization
        # perform the round and get the history of the users
        users_history, round_profit = env.round(best_allocation)
        # Update graph probabilities according to users history
        graph_estimator.update_graph_probabilities(users_history)
        estimator.update_graph_clicks(graph_estimator.get_estimated_graph())
        print(best_allocation)

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
    [
    "experiments: "+str(N_EXPERIMENTS),
    "rounds: "+str(TIME_HORIZON),
    "regret", list(np.mean(mean_regret, axis=0)), 
    "profit", list(np.mean(mean_profit, axis=0)),
    "std_dev", list(np.std(mean_profit, axis=0))]
    )

plt.figure(0)
plt.ylim(-2, 35000)
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot(np.mean(mean_regret, axis=0), 'r')

plt.legend(["REGRET"])
plt.show()

plt.figure(1)
plt.ylim(-2, 35000)
plt.ylabel("Profit")
plt.xlabel("t")
plt.plot(np.mean(mean_profit, axis=0), 'g')
plt.axhline(y=best_expected_profit, color='b', linestyle='-')

plt.legend(["PROFIT", "OPTIMAL AVG"])
plt.show()

if __name__ == '__main__':
    print("simulation done")
