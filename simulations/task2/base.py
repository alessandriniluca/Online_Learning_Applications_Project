import numpy as np

from common.utils import load_static_configuration
from environment.environment import Environment
from optimizer.estimator import Estimator

np.set_printoptions(formatter={'float': lambda x: "{0:0.10f}".format(x)})


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

buy_probs = estimator.get_buy_probs()

print(buy_probs)

if __name__ == '__main__':
    print("simulation done")
