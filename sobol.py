from scipy.stats import qmc
import matplotlib.pyplot as plt

l_bounds = [0.2, 0.2]
u_bounds = [0.8, 0.8]
sampler = qmc.Sobol(d=2, seed=42)
samples = sampler.random_base2(m=3)
samples = qmc.scale(samples, l_bounds, u_bounds)

finalSamples = []
for c, m in samples:
    finalSamples.append([round(c, 2), round(m, 2)])

points = finalSamples

x = [point[0] for point in points]
y = [point[1] for point in points]

plt.scatter(x, y)
plt.xlabel('mutation rate')
plt.ylabel('crossover rate')
plt.title('Sobol Sampling')
print(finalSamples)
plt.show()

