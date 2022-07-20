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

T = 100
n_experiments = 1
n_campaigns = 5

n_arms = int(sim_configuration["total_budget"] / sim_configuration["resolution"]) + 1
budgets = np.linspace(0, sim_configuration["total_budget"], n_arms)

for e in range(0, n_experiments):
    # Initialize a bandits to estimate alpha functions
    gpucb_learners = MultiLearner(n_arms, budgets, LearnerType.UCB1, n_learners=n_campaigns)
    gpts_learners = MultiLearner(n_arms, budgets, LearnerType.TS, n_learners=n_campaigns)

    # TODO note that GPUCB needs to pull every arm when started... combinatorial constraints need to be considered

    # Ask for estimations (get alpha primes

    # Run optimization

    # Compute Rewards from the environment

    # Update the learners



if __name__ == '__main__':
    print("simulation done")
