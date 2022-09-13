from unittest import result
import numpy as np
from matplotlib import pyplot as plt
import sys
import json

if len(sys.argv) <= 2:
    print("USAGE: ")
    print("python3 run.py <result_json> <title>")
    print(" ")
    print("Example task 3 with UCB: python3 run.py results/task_3_ucb.json \"TASK 3 UCB\"")
    print(" ")
    exit(1)

json_filename = sys.argv[1]
title = sys.argv[2]

json_file = open(json_filename)
result_data = json.load(json_file)

mean_profit = np.array(result_data['profit_means'])
mean_profit_std_dev = np.array(result_data['profit_means_std_dev'])
mean_regret = np.array(result_data['regret_means'])
best_expected_profit = result_data['best_expected_profit']
TIME_HORIZON = result_data['rounds']


# #################################### #
# ########## REGRET PLOT ############# #
# #################################### #
plt.figure(0)
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot(mean_regret, 'r')

plt.ylim(bottom=-1000)

plt.axhline(y=0, color='k')

plt.legend(["REGRET"])

plt.title(title + "\nREGRET PLOT")

plt.savefig(json_filename + "_REGRET__.png", dpi=300)

# #################################### #
# #################################### #
# #################################### #


# #################################### #
# ########## PROFIT PLOT ############# #
# #################################### #

plt.figure(1)
plt.ylabel("Profit")
plt.xlabel("t")
plt.plot(mean_profit, 'g')
plt.axhline(y=best_expected_profit, color='b', linestyle='-')

x = np.linspace(0, TIME_HORIZON-1, TIME_HORIZON)

plt.fill_between(x, mean_profit - mean_profit_std_dev, mean_profit + mean_profit_std_dev, color='C0', alpha=0.2)


plt.legend(["PROFIT", "CLAIRVOYANT AVG", "PROFIT STD DEV"])

plt.title(title + "\nREWARD PLOT and CLAIRVOYANT PLOT")

plt.savefig(json_filename + "_PROFIT__.png", dpi=300)


# #################################### #
# #################################### #
# #################################### #



plt.show()

