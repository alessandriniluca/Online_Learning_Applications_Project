import numpy as np

from common.utils import load_static_configuration, get_logger
from environment.environment import Environment
from optimizer.estimator import Estimator
from optimizer.mc_estimator import Graph
from tqdm.auto import tqdm

np.set_printoptions(formatter={'float': lambda x: "{0:0.10f}".format(x)})

logger = get_logger(__name__)

logger.info("Starting Simulation")

configuration = load_static_configuration("../../configurations/static_conf_1.json")

env = Environment(
    configuration=configuration
)

estimator = Estimator(env.configuration.graph_clicks,
                      env.products,
                      env.configuration.lambda_prob,
                      env.configuration.reservation_price_means,
                      env.configuration.reservation_price_std_dev
                      )

logger.debug("Direct estimate computation START")
buy_probs = estimator.get_buy_probs()
logger.debug("Direct estimate computation END")

logger.debug("MC estimate computation START")
buy_probs_mc = []
for class_index in tqdm(range(len(env.configuration.reservation_price_means))):
    buy_probs_current_class = []
    for product_index in range(len(env.products)):
        g = Graph(
            click_probabilities=env.configuration.graph_clicks,
            products=env.products,
            reservation_price_means=env.configuration.reservation_price_means[class_index],
            reservation_price_std_dev=env.configuration.reservation_price_std_dev[class_index],
            lambda_prob=env.configuration.lambda_prob
        )
        g.simulate(starting_product_number=product_index, spin=100000)

        # add current product estimates
        buy_probs_current_class.append(g.get_results())

    buy_probs_mc.append(buy_probs_current_class)
logger.debug("MC estimate computation END")

logger.info("Comparing results")
check_passed = True
for class_index in range(len(env.configuration.reservation_price_means)):
    for starting_product in range(len(env.products)):
        for target_product in range(len(env.products)):
            residual = buy_probs[class_index][starting_product][target_product] - buy_probs_mc[class_index][starting_product][target_product]
            if residual ** 2 > 0.01:
                check_passed = False

if check_passed:
    print("Estimation and exact values matches")


