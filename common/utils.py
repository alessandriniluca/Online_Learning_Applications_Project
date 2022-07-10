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

def load_static_configuration(path: str):
    logger.debug("Loading configuration")
    # Check path
    if not exists(path):
        raise ValueError("Configuration file not found")

    f = open(path)
    parameters = json.load(f)

    configuration = Configuration(
        parameters['average_users_number'],
        parameters['std_users'],
        parameters['basic_alphas'],
        parameters['alphas_functions_parameters'],
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


def basic_exponential_alpha_function(parameters, budget):
    return (parameters[0] * (1.0 - np.exp(parameters[1] * budget))).astype(int)


def alphas_function_for_one_class(parameters_list, budgets):
    assert len(parameters_list) == len(budgets)
    increment = []
    for i, budget in enumerate(budgets):
        increment.append(basic_exponential_alpha_function(parameters_list[i], budget))
    return increment


def get_alpha_functions(parameters):
    lambdas = []
    for user_class_parameters in parameters:
        lambdas.append(lambda x: alphas_function_for_one_class(user_class_parameters, x))
    return lambdas


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