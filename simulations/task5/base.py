import numpy as np

from common.utils import load_static_env_configuration, load_static_sim_configuration, get_test_alphas_functions
from environment.configuration import Configuration
from environment.environment_complete_history import EnvironmentCompleteHistory
from optimizer.estimator import Estimator
from optimizer.full_optimizer import FullOptimizer
from optimizer.optimizer import Optimizer
from probability_calculator.uncertain_graph_weights_estimator import GraphWeightsEstimator
from common.utils import get_logger, get_products

np.set_printoptions(formatter={'float': lambda x: "{0:0.10f}".format(x)})

logger = get_logger(__name__)


env_configuration = load_static_env_configuration("../../configurations/environment/static_conf_1.json")
sim_configuration = load_static_sim_configuration("../../configurations/simulation/sim_conf_1.json")
alphas_functions = get_test_alphas_functions()

ROUNDS = 90
N_EXPERIMENTS = 3
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


for experiment in range(N_EXPERIMENTS):
    for round in range(ROUNDS):

        #set probabilities
        buy_probs = estimator.get_buy_probs()
        optimizer.set_buy_probabilities(buy_probs)

        #Optimize with probabilities calculated
        optimizer.one_campaign_per_product = True
        optimizer.run_optimization()
        best_allocation, expected_earning = optimizer.find_best_allocation()
        # end of optimization
        # perform the round and get the history of the users
        users_history = env.round(best_allocation)
        # Update graph probabilities according to users history
        graph_estimator.update_graph_probabilities(users_history)
        estimator.update_graph_clicks(graph_estimator.get_estimated_graph())
        print(best_allocation)
    # For each experiment, we have to zero everything to repeat the estimation
    test_experiments.append(graph_estimator.get_estimated_graph().copy())
    graph_estimator.reset()
    optimizer.set_buy_probabilities(estimator.get_buy_probs())
    logger.info("Best allocation for experiment: " + str(best_allocation))
    logger.info("Reward: ")

print("\n\n\n")
for i, e in enumerate(test_experiments):
    logger.info("\n Experiment numeber " + str(i) + ". Results:\n" + str(e))

if __name__ == '__main__':
    logger.info("simulation done")
