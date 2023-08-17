import itertools

from scipy.stats import qmc
import numpy as np

gamma_range = [round(i * 0.2, 1) for i in range(0, 6)]
iterationsPerEnv_range = [25, 50, 75, 100, 150, 250]
crossover_prob_range = [round(i * 0.2, 1) for i in range(0, 6)]
mutation_prob_range = [round(i * 0.2, 1) for i in range(0, 6)]

# Define the parameters and their ranges
param_ranges = {
    "numCurric": [1, 5],
    "curricLen": [2, 5],
    "nGen": [1, 5],
    "paraEnv": [1, 2],
    "gamma": [min(gamma_range), max(gamma_range)],
    "iterationsPerEnv": [25, 250],
    "crossover": [min(crossover_prob_range), max(crossover_prob_range)],
    "mutation": [min(mutation_prob_range), max(mutation_prob_range)]
}

# Number of samples to generate
n_samples = 16

# Create a Sobol sequence instance
sampler = qmc.Sobol(len(param_ranges))

# Generate the samples
samples = sampler.random(n_samples)

# Scale the values to the right range
scaled_samples = []
for sample in samples:
    scaled_sample = {}
    for i, (param_name, param_range) in enumerate(param_ranges.items()):
        lower, upper = param_range
        value = sample[i] * (upper - lower) + lower

        # Round to nearest integer if the range is integer
        if isinstance(lower, int) and isinstance(upper, int):
            value = int(round(value))

        # For iterationsPerEnv, round to nearest 25000
        if param_name == "iterationsPerEnv":
            print(value)
            value = int(round(value / 25) * 25)

        scaled_sample[param_name] = value

    scaled_samples.append(scaled_sample)
# Print the hyperparameter configurations
for i, sample in enumerate(scaled_samples):
    print(f"Hyperparameter Configuration {i+1}:", end=" ")
    for param_name, value in sample.items():
        print(f"{param_name}: {value}", end="; ")
    print()

exit()

# Define hyperparameters ranges
numCurric_range = range(1, 6)
curricLen_range = range(2, 6)
nGen_range = range(1, 6)
paraEnv_range = range(1, 3)
gamma_range = [round(i * 0.2, 1) for i in range(0, 6)]
iterationsPerEnv_range = [25, 50, 75, 100, 150, 250]
crossover_prob_range = [round(i * 0.2, 1) for i in range(0, 6)]
mutation_prob_range = [round(i * 0.2, 1) for i in range(0, 6)]     #

# generate grid using itertools
grid = list(itertools.product(numCurric_range, curricLen_range, nGen_range, paraEnv_range, gamma_range, iterationsPerEnv_range, crossover_prob_range, mutation_prob_range ))

print("Number of combination generated:", len(grid))
print("First 5 combinations:")
for grid_item in grid[:5]:
    print(grid_item)


