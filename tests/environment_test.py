from common.utils import load_static_configuration
from environment.environment import Environment

configuration = load_static_configuration("../configurations/static_conf_1.json")

test_env = Environment(
    configuration=configuration
)

test_env.round([0, 0, 0, 0, 0])

if __name__ == '__main__':
    print("ok")
