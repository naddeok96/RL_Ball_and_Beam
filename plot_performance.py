
import matplotlib.pyplot as plt
import numpy as np

performance = np.loadtxt('q_table_small2_performance.csv', delimiter=',')

plt.plot(performance)
plt.xlabel("Intervals")
plt.ylabel("Percent of Successes in Interval")

plt.show()