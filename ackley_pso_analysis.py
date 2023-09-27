import numpy as np
import matplotlib.pyplot as plt
from pso import PSOOptimizer
from Functions import AckleyFunction

dimensions = 10
bounds = (-30, 30)
iterations = 1000
num_runs = 10

ackley_function = AckleyFunction(dimensions)

results = []
for _ in range(num_runs):
    pso_optimizer = PSOOptimizer(ackley_function.evaluate, dimensions, bounds, max_iterations=iterations)
    _, best_value = pso_optimizer.optimize()
    results.append(best_value)

plt.plot(results, marker='o')
plt.title('PSO Runs on Ackley Function')
plt.xlabel('Run')
plt.ylabel('Best Value')
plt.grid(True)
plt.savefig('results/metrics std pso.png')
plt.show()

with open('results/analysis std pso.txt', 'w') as f:
    f.write("Analysis of PSO runs on Ackley Function:\n")
    f.write(f"Average best value: {np.mean(results)}\n")
    f.write(f"Standard deviation: {np.std(results)}\n")
    
print("completed")