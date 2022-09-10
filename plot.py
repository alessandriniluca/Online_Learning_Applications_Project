from unittest import result
import numpy as np
from matplotlib import pyplot as plt
import sys
import json

if len(sys.argv) <= 1:
    print("USAGE: ")
    print("python3 run.py <result_json>")
    print(" ")
    print("Example task 2: python3 run.py results/<aaa>.json")
    print(" ")
    exit(1)

json_filename = sys.argv[1]

json_file = open(json_filename)
result_data = json.load(json_file)

mean_profit = np.array(result_data['profit_means'])
mean_profit_std_dev = np.array(result_data['profit_means_std_dev'])
mean_regret = np.array(result_data['regret_means'])
best_expected_profit = result_data['best_expected_profit']
TIME_HORIZON = result_data['rounds']

plt.figure(0)
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot(mean_regret, 'r')

plt.legend(["REGRET"])
plt.show()

plt.figure(1)
plt.ylabel("Profit")
plt.xlabel("t")
plt.plot(mean_profit, 'g')
plt.axhline(y=best_expected_profit, color='b', linestyle='-')

x = np.linspace(0, TIME_HORIZON-1, TIME_HORIZON)

plt.fill_between(x, mean_profit - mean_profit_std_dev, mean_profit + mean_profit_std_dev, color='C0', alpha=0.2)


plt.legend(["PROFIT", "OPTIMAL AVG"])
plt.show()

