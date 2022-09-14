from ast import If
from unittest import result
import numpy as np
from matplotlib import pyplot as plt
import sys
import json

if len(sys.argv) < 3:
    print("USAGE: ")
    print("python3 run.py <result_json> <title>")
    print(" ")
    print("Example task 3 with UCB: python3 run.py results/task_3_ucb.json \"TASK 3 UCB\"")
    print(" ")
    exit(1)

context = False
if len(sys.argv) > 3:
    if sys.argv[3] == "CONTEXT":
            context = True

json_filename = sys.argv[1]
title = sys.argv[2]

json_file = open(json_filename)
result_data = json.load(json_file)

mean_profit = np.array(result_data['profit_means'])
mean_profit_std_dev = np.array(result_data['profit_means_std_dev'])
mean_regret = np.array(result_data['regret_means'])
best_expected_profit = mean_profit + mean_regret #result_data['best_expected_profit']
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
plt.plot(best_expected_profit, 'b')

x = np.linspace(0, TIME_HORIZON-1, TIME_HORIZON)

plt.fill_between(x, mean_profit - mean_profit_std_dev, mean_profit + mean_profit_std_dev, color='C0', alpha=0.2)

if context:
    for i in range(len(mean_profit)):
        if i % 14 == 0:
            plt.axvline(x=i, color='r')

    plt.legend(["PROFIT", "CLAIRVOYANT AVG", "PROFIT STD DEV", "CONTEXT SPLIT"])
else:
    plt.legend(["PROFIT", "CLAIRVOYANT AVG", "PROFIT STD DEV"])

plt.title(title + "\nREWARD PLOT and CLAIRVOYANT PLOT")

plt.savefig(json_filename + "_PROFIT__.png", dpi=300)


# #################################### #
# #################################### #
# #################################### #



plt.show()

