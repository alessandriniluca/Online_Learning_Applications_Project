import enum
import logging
from os.path import exists
import json
import numpy as np

from environment.configuration import Configuration
from environment.product import Product

LOG_FORMAT = "[%(levelname)s] [%(name)s] [%(asctime)s] %(message)s"

logging.basicConfig(
    format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)


def get_logger(file_name: str) -> logging.Logger:
    """
    Return logger with name and level given in input. Call should be:
    my_logger = get_logger(__name__)
    Parameters
    ----------
        - file_name: name of the logger. It should be the name of the
        current file.
    """

    logger = logging.getLogger(file_name)
    logger.setLevel(logging.DEBUG)  # the level is fixed to DEBUG
    return logger


logger = get_logger(__name__)


def load_static_env_configuration(path: str):
    logger.debug("Loading environment configuration")
    # Check path
    if not exists(path):
        raise ValueError("Configuration file not found")

    f = open(path)
    parameters = json.load(f)

    configuration = Configuration(
        parameters['average_users_number'],
        parameters['std_users'],
        parameters['basic_alphas'],
        parameters['products_parameters'],
        parameters['lambda_prob'],
        parameters['graph_clicks'],
        parameters['reservation_price_means'],
        parameters['reservation_price_std_dev'],
        parameters['quantity_means'],
        parameters['quantity_std_dev']
    )

    # Closing file
    f.close()

    return configuration


def get_products(parameters):
    products = []
    for product_param in parameters:
        products.append(
            Product(
                name=product_param["name"],
                price=product_param["price"],
                number=product_param["number"]
            )
        )

    for i, product in enumerate(products):
        product.set_secondary(
            products[parameters[i]["secondary"][0]],
            products[parameters[i]["secondary"][1]]
        )

    return products


def load_static_sim_configuration(path: str):
    logger.debug("Loading simulation configuration")
    # Check path
    if not exists(path):
        raise ValueError("Configuration file not found")

    f = open(path)
    configuration = json.load(f)
    return configuration


def alphas_function_class_0(budget):
    increment = [(30.0 * (1.0 - np.exp(-0.04 * (budget[0])))).astype(int),
                 (10.0 * (1.0 - np.exp(-0.035 * (budget[1])))).astype(int),
                 (15.0 * (1.0 - np.exp(-0.045 * (budget[2])))).astype(int),
                 (25.0 * (1.0 - np.exp(-0.04 * (budget[3])))).astype(int),
                 (60.0 * (1.0 - np.exp(-0.05 * (budget[4])))).astype(int)]
    return np.array(increment)


def alphas_function_class_1(budget):
    increment = [(25.0 * (1.0 - np.exp(-0.043 * (budget[0])))).astype(int),
                 (15.0 * (1.0 - np.exp(-0.039 * (budget[1])))).astype(int),
                 (10.0 * (1.0 - np.exp(-0.045 * (budget[2])))).astype(int),
                 (10.0 * (1.0 - np.exp(-0.03 * (budget[3])))).astype(int),
                 (70.0 * (1.0 - np.exp(-0.029 * (budget[4])))).astype(int)]

    return np.array(increment)


def alphas_function_class_2(budget):
    increment = [(30.0 * (1.0 - np.exp(-0.038 * (budget[0])))).astype(int),
                 (10.0 * (1.0 - np.exp(-0.045 * (budget[1])))).astype(int),
                 (50.0 * (1.0 - np.exp(-0.045 * (budget[2])))).astype(int),
                 (15.0 * (1.0 - np.exp(-0.044 * (budget[3])))).astype(int),
                 (45.0 * (1.0 - np.exp(-0.043 * (budget[4])))).astype(int)]
    return np.array(increment)


def get_test_alphas_functions():
    return [alphas_function_class_0, alphas_function_class_1, alphas_function_class_2]


def get_prices(products):
    prices = []
    for product in products:
        prices.append(product.price)
    return prices


class LearnerType(enum.Enum):
    """
    Learner types
    """

    TS = enum.auto()
    UCB1 = enum.auto()
